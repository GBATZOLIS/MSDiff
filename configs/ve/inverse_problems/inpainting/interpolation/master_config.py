from configs.ve.inverse_problems.inpainting.interpolation import c1,c2,c3,c4,c5,c6,c7,c8,c9,c10
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.c1 = c1.get_config()
    config.c2 = c2.get_config()
    config.c3 = c3.get_config()
    config.c4 = c4.get_config()
    config.c5 = c5.get_config()
    config.c6 = c6.get_config()
    config.c7 = c7.get_config()
    config.c8 = c8.get_config()
    config.c9 = c9.get_config()
    config.c10 = c10.get_config()
    return config