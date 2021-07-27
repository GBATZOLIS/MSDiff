#%%
import torch
import pytorch_lightning as pl
from sklearn import datasets
from matplotlib import pyplot as plt
from lightning import SdeGenerativeModel
from models import ddpm, ncsnv2, fcn
from configs.vp.toy_moons import get_config
config = get_config()
#%%
data, labels = datasets.make_moons(n_samples=config.data.data_samples,
                                   shuffle=False, 
                                   noise=config.data.noise_scale, 
                                   random_state=0)
plt.scatter(data[:,0], data[:,1])
#%%
data_tensor = torch.from_numpy(data).float()
labels_tensor = torch.from_numpy(labels).float()

dataset=torch.utils.data.TensorDataset(data_tensor, labels_tensor)
batch_size=config.training.batch_size
data_loader=torch.utils.data.DataLoader(dataset, 
                                        batch_size=batch_size,
                                        num_workers=4)

# %%
model = SdeGenerativeModel(config)
# %%
trainer = pl.Trainer(gpus=1,max_epochs=5)
# %%
trainer.fit(model, data_loader)
# %%
import sampling
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)

shape = (batch_size, 2)
predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = None #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = config.sampling.snr #@param {"type": "number"}
n_steps =  config.sampling.n_steps_each #@param {"type": "integer"}
probability_flow = config.sampling.probability_flow #@param {"type": "boolean"}
sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
sampling_eps = 1e-3
sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)
#%%
model.to(config.device)
x, n = sampling_fn(model.score_model)
#%%
x_np = x.cpu().numpy()
plt.scatter(x_np[:,0], x_np[:,1])
# %%
