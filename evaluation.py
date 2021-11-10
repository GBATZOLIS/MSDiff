#This file is used for evaluation.
#We construct the dataloader and the entire evaluation pipeline.

from torch.utils.data import  Dataset, DataLoader
import glob
import os 
from torchvision.transforms import ToTensor
from PIL import Image
import lpips
from lightning_callbacks.evaluation_tools import calculate_mean_psnr, calculate_mean_ssim, get_calculate_consistency_fn
import numpy as np
import pickle 
import torch
from tqdm import tqdm
from scipy import linalg
import copy

#for the fid calculation
from models.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d

def listdir_nothidden_paths(path, filetype=None):
    if not filetype:
        return glob.glob(os.path.join(path, '*'))
    else:
        return glob.glob(os.path.join(path, '*.%s' % filetype))


def listdir_nothidden_filenames(path, filetype=None):
    if not filetype:
        paths = glob.glob(os.path.join(path, '*'))
    else:
        paths = glob.glob(os.path.join(path, '*.%s' % filetype))
    
    files = [os.path.basename(path) for path in paths]
    return files


def get_gt_draw_to_file_fn(gt_draw_files): #some draws share the same ground truths.
    draw_to_file_dict = {}
    for draw_file in gt_draw_files:
        if len(draw_file.split('_')) == 2:
            draw_to_file_dict[int(draw_file.split('_')[1])]=draw_file
        elif len(draw_file.split('_')) == 3:
            start = int(draw_file.split('_')[1])
            end = int(draw_file.split('_')[2])
            for i in range(start, end+1):
                draw_to_file_dict[i]=draw_file

    def draw_to_file_fn(draw : int):
        return draw_to_file_dict[draw]
    
    return draw_to_file_fn

class SynthesizedDataset(Dataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self, task, base_path, snr):
        #base_path: -> samples -> snr_0.150 -> draw_i
        #           -> gt -> draw_i_j or draw_i -> x_gt
        #           -> gt -> draw_i_j or draw_i -> y_gt

        self.task = task #use for determining the extra information (usually details related to the forward operator)

        self.sample_paths = {}
        base_sample_path = os.path.join(base_path, 'samples', 'snr_%.3f' % snr)
        
        self.gt_paths = {'x':{}, 'y':{}}
        base_gt_path = os.path.join(base_path, 'gt')
        gt_draw_files = listdir_nothidden_filenames(base_gt_path)

        gt_draw_to_file_fn = get_gt_draw_to_file_fn(gt_draw_files)

        self.draw_paths = listdir_nothidden_filenames(base_sample_path)
        for draw_path in self.draw_paths:
            draw = int(draw_path.split('_')[1])
            self.sample_paths[draw] = sorted(listdir_nothidden_paths(os.path.join(base_sample_path, draw_path), 'png'))
            self.gt_paths['x'][draw] = sorted(listdir_nothidden_paths(os.path.join(base_gt_path, gt_draw_to_file_fn(draw), 'x_gt'), 'png'))
            self.gt_paths['y'][draw] = sorted(listdir_nothidden_paths(os.path.join(base_gt_path, gt_draw_to_file_fn(draw), 'y_gt'), 'png'))

    def __getitem__(self, index):        
        gt_y = {}
        gt_x = {}
        samples = {}
        for draw in self.sample_paths.keys():
            samples[draw] = ToTensor()(Image.open(self.sample_paths[draw][index]).convert('RGB'))
            gt_y[draw]= ToTensor()(Image.open(self.gt_paths['y'][draw][index]).convert('RGB'))
            gt_x[draw]= ToTensor()(Image.open(self.gt_paths['x'][draw][index]).convert('RGB'))
            
        info = {'y': gt_y,
                'samples': samples,
                'x': gt_x}

        if self.task == 'inpainting':
            mask_coverage = 0.25
            a_sample = samples[list(samples.keys())[0]]
            size_x, size_y = a_sample.shape[1], a_sample.shape[2]
            mask_size = int(np.sqrt(mask_coverage * size_x * size_y))
            start_x = np.random.randint(low=0, high=(size_x - mask_size) + 1) if size_x > mask_size else 0
            start_y = np.random.randint(low=0, high=(size_y - mask_size) + 1) if size_y > mask_size else 0

            mask_info_tensor = torch.tensor([start_x, start_y, mask_size], dtype=torch.int32)

            info['mask_info']={}
            for draw in self.sample_paths.keys():   
                info['mask_info'][draw] = mask_info_tensor

        return info
    
    def __len__(self):
        min_draws = min([len(self.sample_paths[draw]) for draw in self.sample_paths.keys()])
        return min_draws

def get_activation_fn(model):
    def activation_fn(img):
        with torch.no_grad():
            pred = model(img)[0]
        
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        activation = pred.squeeze(3).squeeze(2).cpu()
        return activation
    return activation_fn

def get_fid_fn(distribution):
    if distribution == 'target': #unconditional fid
        def fid_fn(acts):
            activations = copy.deepcopy(acts)
            target_act_stats = {}
            sample_act_stats = {}
            target_fid = {}
            for draw in activations['samples'].keys():
                sample_activations = torch.cat(activations['samples'][draw], dim=0).numpy()
                sample_act_stats[draw] = {'mu':np.mean(sample_activations, axis=0), 'sigma':np.cov(sample_activations, rowvar=False)}
                
                target_activations = torch.cat(activations['x'][draw], dim=0).numpy()
                target_act_stats[draw] = {'mu':np.mean(target_activations, axis=0), 'sigma':np.cov(target_activations, rowvar=False)}
                
                mu1, sigma1 = target_act_stats[draw]['mu'], target_act_stats[draw]['sigma']
                mu2, sigma2 = sample_act_stats[draw]['mu'], sample_act_stats[draw]['sigma']
                target_fid[draw] = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            
            del activations
            return target_fid

    elif distribution == 'joint': #joint fid
        def fid_fn(acts):
            activations = copy.deepcopy(acts)
            activations_y_x = {}
            activations_y_samples = {}
            for draw in activations['samples'].keys():
                activations_y_x[draw]=[]
                activations_y_samples[draw]=[]

            num_images = len(activations['samples'][list(activations['samples'].keys())[0]])
            for i in range(num_images):
                for draw in activations['samples'].keys():
                    concat_act_y_sample = torch.cat((activations['y'][draw][i], activations['samples'][draw][i]), dim=-1)
                    activations_y_samples[draw].append(concat_act_y_sample)

                    concat_act_y_x = torch.cat((activations['y'][draw][i], activations['x'][draw][i]), dim=-1)
                    activations_y_x[draw].append(concat_act_y_x)
            
            joint_fid = {}
            for draw in activations['samples'].keys():
                activations_y_x_draw = torch.cat(activations_y_x[draw], dim=0).numpy()
                gt_draw_stats = {'mu':np.mean(activations_y_x_draw, axis=0),
                                 'sigma':np.cov(activations_y_x_draw, rowvar=False)}

                activations_y_samples_draw = torch.cat(activations_y_samples[draw], dim=0).numpy()
                sample_draw_stats = {'mu':np.mean(activations_y_samples_draw, axis=0), 
                                     'sigma':np.cov(activations_y_samples_draw, rowvar=False)}
                
                mu1, sigma1 = gt_draw_stats['mu'], gt_draw_stats['sigma']
                mu2, sigma2 = sample_draw_stats['mu'], sample_draw_stats['sigma']
                joint_fid[draw] = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            
            del activations
            return joint_fid
    
    return fid_fn

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def run_evaluation_pipeline(task, base_path, snr, device):
    #report: 
    #1.) Expected LPIPS
    #2.) Expected PSNR
    #3.) Expected SSIM
    #4.) Expected consistency #(make this general) -> to be done
    #5.) Expected FID (unconditional) ##to be done
    #6.) Expected Joint FID (conditional) ##to be done
    #7.) The identification info of the 20 best samples
    #    based on the lowest LPIPS scores. (report image name + draw)

    #set up the inception model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx], resize_input=True).to(device)
    inception_model.eval()
    activation_fn = get_activation_fn(inception_model)

    dataset = SynthesizedDataset(task, base_path, snr)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle=False, num_workers=8) 

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    consistency_fn = get_calculate_consistency_fn(task)

    lpips_val_to_imgID = {}
    all_lpips_values = []
    
    mean_lpips_values = []
    mean_psnr_values = []
    mean_ssim_values = []
    mean_consistency_values = []
    diversities = []
    activations = {'x':{},
                   'y':{},
                   'samples': {}}

    for i, info in tqdm(enumerate(dataloader)):
        if i>10:
            break
        y, x = info['y'], info['x']
        samples = info['samples']

        if i==0: #populate the empty activation dictionaries with the draw keys.
            for draw in samples.keys():
                activations['samples'][draw]=[]
                activations['x'][draw]=[]
                activations['y'][draw]=[]

        lpips_values = []
        psnr_values = []
        ssim_values = []
        consistency_values = []
        
        concat_samples = [] #use for calculating diversity
        
        for draw in samples.keys():
            y[draw] = y[draw].to(device)
            x[draw] = x[draw].to(device)
            samples[draw] = samples[draw].to(device)

            #FID
            #calculate the inception activation for the gt and synthetic samples.
            activations['y'][draw].append(activation_fn(y[draw].to(device)))
            activations['x'][draw].append(activation_fn(x[draw].to(device)))
            activations['samples'][draw].append(activation_fn(samples[draw].to(device)))
            
            #LPIPS
            lpips_val = loss_fn_alex(2*x[draw].clone()-1, 2*samples[draw].clone()-1).cpu().squeeze().item()
            if lpips_val in lpips_val_to_imgID.keys():
                lpips_val_to_imgID[lpips_val].extend([(i+1, draw)])
            else:
                lpips_val_to_imgID[lpips_val]=[(i+1, draw)]
            lpips_values.append(lpips_val)
            all_lpips_values.append(lpips_val)
            
            #PSNR, SSIM
            #convert the torch tensors to numpy arrays for the remaining metric calculations
            numpy_samples = torch.swapaxes(samples[draw].clone().cpu(), axis0=1, axis1=-1).numpy()*255
            numpy_gt = torch.swapaxes(x[draw].clone().cpu(), axis0=1, axis1=-1).numpy()*255
            
            psnr_val = calculate_mean_psnr(numpy_samples, numpy_gt)
            psnr_values.append(psnr_val)

            ssim_val = calculate_mean_ssim(numpy_samples, numpy_gt)
            ssim_values.append(ssim_val)
            
            #CONSISTENCY
            if task == 'super-resolution':
                consistency_val = consistency_fn(samples[draw], x[draw], scale=8)
                consistency_values.append(consistency_val)
            elif task == 'inpainting':
                consistency_val = consistency_fn(samples[draw], x[draw], mask_info=info['mask_info'][draw])
            elif task == 'image-to-image':
                consistency_val = consistency_fn(numpy_samples, numpy_gt)

            #DIVERSITY
            if len(samples.keys())>1:
                to_be_concatenated = samples[draw]*255.
                concat_samples.append(to_be_concatenated.cpu())
        
        mean_lpips_value = np.mean(lpips_values)
        mean_psnr_value = np.mean(psnr_values)
        mean_ssim_value = np.mean(ssim_values)
        mean_consistency_value = np.mean(consistency_values)
        

        mean_lpips_values.append(mean_lpips_value)
        mean_psnr_values.append(mean_psnr_value)
        mean_ssim_values.append(mean_ssim_value)
        mean_consistency_values.append(mean_consistency_value)

        if len(samples.keys())>1:
            diversity = torch.mean(torch.std(torch.stack(concat_samples), dim=0)).item()
            diversities.append(diversity)

    #Calculate mean joint and target FID scores.
    joint_fid_fn = get_fid_fn(distribution='joint')
    target_fid_fn = get_fid_fn(distribution='target')

    target_fid_dict = target_fid_fn(activations)
    joint_fid_dict = joint_fid_fn(activations)

    target_fid = {}
    target_fid_values = [target_fid_dict[draw] for draw in target_fid_dict.keys()]
    target_fid['mean'], target_fid['std'] = np.mean(target_fid_values), np.std(target_fid_values)

    joint_fid = {}
    joint_fid_values = [joint_fid_dict[draw] for draw in joint_fid_dict.keys()]
    joint_fid['mean'], joint_fid['std'] = np.mean(joint_fid_values), np.std(joint_fid_values)

    #Calculate the mean values (LPIPS, PSNR, SSIM, CONSISTENCY, DIVERSITY)
    mean_lpips = np.mean(mean_lpips_values)
    mean_psnr = np.mean(mean_psnr_values)
    mean_ssim = np.mean(mean_ssim_values)
    mean_consistency = np.mean(mean_consistency_values)
    mean_diversity = np.mean(diversities)

    #get the id info of the best samples based on LPIPS.
    lpips_values = sorted(all_lpips_values) #increasing order
    best_lpips_samples_id_info = []
    for lpips_val in lpips_values[:20]:
        best_lpips_samples_id_info.extend(lpips_val_to_imgID[lpips_val])

    info = {'lpips':mean_lpips,
            'psnr': mean_psnr,
            'ssim': mean_ssim,
            'consistency': mean_consistency,
            'diversity': mean_diversity,
            'target_fid': target_fid['mean'],
            'target_fid_std': target_fid['std'],
            'joint_fid': joint_fid['mean'],
            'joint_fid_std': joint_fid['std'],
            'best_lpips_samples': best_lpips_samples_id_info}
    
    for key in info.keys():
        if key != 'best_lpips_samples':
            print('%s: %.5f' % (key, info[key]))
        else:
            print('------')
            print('%s' % key)
            print('------')
            for ID in info[key]:
                print(ID)

    f = open(os.path.join(base_path, 'evaluation_info.pkl'), "wb")
    pickle.dump(info, f)
    f.close()
