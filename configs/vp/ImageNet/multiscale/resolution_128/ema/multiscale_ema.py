from configs.vp.ImageNet.multiscale.resolution_128.ema import d1_ema, d2, d3, a3

import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.d1 = d1_ema.get_config()
    config.d2 = d2.get_config()
    config.d3 = d3.get_config()
    config.a3 = a3.get_config()

    return config