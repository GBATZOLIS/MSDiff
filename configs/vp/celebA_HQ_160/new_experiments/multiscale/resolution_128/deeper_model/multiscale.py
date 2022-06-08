from configs.vp.celebA_HQ_160.new_experiments.multiscale.resolution_128.deeper_model import d1, d2, d3, a3

import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.d1 = d1.get_config()
    config.d2 = d2.get_config()
    config.d3 = d3.get_config()
    config.a3 = a3.get_config()

    return config