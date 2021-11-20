from configs.ve.inverse_problems.image_to_image_translation import edges2shoes_ours_DV, edges2shoes_ours_NDV, edges2shoes_song, edges2shoes_SR3
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.ours_DV = edges2shoes_ours_DV.get_config()
    #config.ours_NDV = edges2shoes_ours_NDV.get_config()
    #config.song = edges2shoes_song.get_config()
    #config.SR3 = edges2shoes_SR3.get_config()
    return config