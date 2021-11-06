#This file is used for evaluation.
#We construct the dataloader and the entire evaluation pipeline.

from torch.utils.data import  Dataset, DataLoader
import glob
import os 
from torchvision.transforms import ToTensor
from PIL import Image
import lpips
from lightning_callbacks.evaluation_tools import calculate_mean_psnr, calculate_mean_ssim, get_calculate_consistency_fn

def listdir_nothidden(path, filetype):
    return glob.glob(os.path.join(path, '*.%s' % filetype))

class SynthesizedDataset(Dataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self, base_path, snr):
        #base_path: -> samples -> snr_0.150 -> draw_i
        #           -> x_gt
        #           -> y_gt

        self.sample_paths = {}
        base_sample_path = os.path.join(base_path, 'samples, snr_%.3f' % snr)
        draw_paths = os.listdir(base_sample_path)
        for draw_path in draw_paths:
            self.sample_paths[int(draw_path.split('_')[1])] = sorted(listdir_nothidden(os.path.join(base_sample_path, draw_path), 'png'))
            print(self.sample_paths[int(draw_path.split('_')[1])][:5])

        self.y_paths = sorted(listdir_nothidden(os.path.join(base_path, 'y_gt'), 'png'))
        print(self.y_paths[0:5])
        
        self.x_paths = sorted(listdir_nothidden(os.path.join(base_path, 'x_gt'), 'png'))
        print(self.x_paths[:5])
    
    def __getitem__(self, index):

        y = ToTensor()(Image.open(self.y_paths[index]).convert('RGB'))
        print('y.min(): %.3f, y.max(): %.3f' % (y.min(), y.max()))
        x = ToTensor(Image.open(self.x_paths[index]).convert('RGB'))

        samples = {}
        for draw in self.sample_paths.keys():
            samples[draw] = ToTensor(Image.open(self.sample_paths[draw][index]).convert('RGB'))
        
        return y, samples, x
    

def run_evaluation_pipeline(task, base_path, snr, device):
    #report: 
    #1.) Expected LPIPS
    #2.) Expected PSNR
    #3.) Expected SSIM
    #4.) Expected consistency
    #5.) Expected FID (unconditional)
    #6.) Expected Joint FID (conditional)
    #7.) The identification info of the 20 best samples
    #    based on the lowest LPIPS scores. (report image name + draw)

    dataset = SynthesizedDataset(base_path, snr)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle=False, num_workers=8) 

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    consistency_fn = get_calculate_consistency_fn(task)

    lpips_val_to_imgID = {}
    all_lpips_values = []
    
    all_psnr_values = []
    all_ssim_values = []
    all_consistency_values = []
    all_diversities = []
    for i, (y, samples, x) in enumerate(dataloader):
        x = x.to(device)
        concat_samples = []
        for draw in samples.keys():
            samples[draw].to(device)
            lpips_val = loss_fn_alex(2*x.clone()-1, 2*samples[draw].clone()-1).cpu().squeeze().item()
            
            if lpips_val in lpips_val_to_imgID.keys():
                lpips_val_to_imgID[lpips_val].extend([(i+1, draw)])
            else:
                lpips_val_to_imgID[lpips_val]=[(i+1, draw)]
            
            all_lpips_values.append(lpips_val)

            #convert the torch tensors to numpy arrays for the remaining metric calculations
            numpy_samples = torch.swapaxes(samples[draw].clone().cpu(), axis0=1, axis1=-1).numpy()*255
            numpy_gt = torch.swapaxes(x.clone().cpu(), axis0=1, axis1=-1).numpy()*255
            
            psnr_val = calculate_mean_psnr(numpy_samples, numpy_gt)
            all_psnr_values.append(psnr_val)

            ssim_val = calculate_mean_ssim(numpy_samples, numpy_gt)
            all_ssim_values.append(ssim_val)
            
            if task == 'super-resolution':
                consistency_val = consistency_fn(samples[draw], x, scale=8)
                all_consistency_values.append(consistency_val)
            
            if len(samples.keys())>1:
                to_be_concatenated = samples[draw]*255.
                concat_samples.append(to_be_concatenated.cpu())
        
        diversity = torch.mean(torch.std(torch.stack(concat_samples), dim=0)).item()
        all_diversities.append(diversity)




    #get the id info of the best samples based on LPIPS.
    lpips_values = sorted(all_lpips_values) #increasing order
    best_lpips_samples_id_info = []
    for lpips_val in lpips_values[:20]:
        best_lpips_samples_id_info.extend(lpips_val_to_imgID[lpips_val])


    
    



