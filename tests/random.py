#%%
import torch
# %%
x = torch.tensor([1,2,3])
# %%
x.shape
# %%
x.expand(-1,1,1,1)
# %%
