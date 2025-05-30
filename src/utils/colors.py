# https://github.com/Co-Evolve/brt/blob/main/biorobot/utils/colors.py

import numpy as np
from PIL import ImageColor

hex_green = "#7db5a8"
hex_gray = "#595959"

rgba_green = np.array(ImageColor.getcolor(hex_green, "RGBA")) / 255
rgba_gray = np.array(ImageColor.getcolor(hex_gray, "RGBA")) / 255
