#%% 
import os
os.chdir('..')
# %% 
from lightning_data_modules.ImageDatasets import ImageDataModule
from torch.utils.data import DataLoader
# %%
train_loader = DataLoader(celebA, 
                            batch_size=128,
                            num_workers=0,
                            shuffle=True,
                            drop_last=True)
# %%
next(iter(train_loader)).shape
# %%
