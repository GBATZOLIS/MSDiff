from configs.ve.inverse_problems.inpainting import celebA_ours_DV, celebA_ours_NDV, celebA_song, celebA_SR3
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.ours_DV = celebA_ours_DV.get_config()
    #config.ours_NDV = celebA_ours_NDV.get_config()
    #config.song = celebA_song.get_config()
    #config.SR3 = celebA_SR3.get_config()
    return config