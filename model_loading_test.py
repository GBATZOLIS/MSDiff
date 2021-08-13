#%%
import pytorch_lightning as pl
import ml_collections
from matplotlib import pyplot as plt
from model_lightning import SdeGenerativeModel
#%%
from configs.vp.SyntheticDataset import get_config
config=get_config()
# %%
path = '/home/js2164/jan/repos/score_sde_pytorch-1/lightning_logs/version_16/checkpoints/epoch=47-step=3792.ckpt'
model = SdeGenerativeModel.load_from_checkpoint(checkpoint_path=path)
# %%
trainer=pl.Trainer(resume_from_checkpoint=path)
# %%
trainer.fit(model)

# %%
