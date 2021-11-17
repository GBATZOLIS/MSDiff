from configs.ve.inverse_problems.super_resolution import deep_celebA_ours_NDV_160, deep_celebA_SR3
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.ours_NDV = deep_celebA_ours_NDV_160.get_config()
    #config.SR3 = deep_celebA_SR3.get_config()
    return config