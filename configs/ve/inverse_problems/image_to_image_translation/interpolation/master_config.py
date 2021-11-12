from configs.ve.inverse_problems.image_to_image_translation.interpolation import ours_NDV_1, ours_NDV_2, ours_NDV_3, ours_NDV_4, ours_NDV_5, ours_NDV_6, ours_NDV_7, ours_NDV_8, ours_NDV_9, SR3
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.ours_DV_1 = ours_NDV_1.get_config()
    config.ours_DV_2 = ours_NDV_2.get_config()
    config.ours_DV_3 = ours_NDV_3.get_config()
    config.ours_DV_4 = ours_NDV_4.get_config()
    config.ours_DV_5 = ours_NDV_5.get_config()
    config.ours_DV_6 = ours_NDV_6.get_config()
    config.ours_DV_7 = ours_NDV_7.get_config()
    config.ours_DV_8 = ours_NDV_8.get_config()
    config.ours_DV_9 = ours_NDV_9.get_config()

    config.SR3 = SR3.get_config()
    return config