import skimage.io as io
from facenet_models import FacenetModel
import numpy as np
model = FacenetModel()

# shape-(Height, Width, Color)
image = io.imread(str(path_to_image))
if image.shape[-1] == 4:
    # Image is RGBA, where A is alpha -> transparency
    # Must make image RGB.
    image = image[..., :-1]  # png -> RGB

for 