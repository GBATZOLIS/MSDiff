from configs.ve.inverse_problems.super_resolution import celebA_ours_DV_160, celebA_ours_slowDV_160, celebA_ours_NDV_160, celebA_song_160, celebA_SR3_160
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.ours_DV = celebA_ours_DV_160.get_config()
    #config.ours_slowDV = celebA_ours_slowDV_160.get_config()
    #config.ours_NDV = celebA_ours_NDV_160.get_config()
    #config.song = celebA_song_160.get_config()
    #config.SR3 = celebA_SR3_160.get_config()
    return config