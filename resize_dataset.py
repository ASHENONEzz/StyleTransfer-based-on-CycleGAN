import numpy as np
from PIL import Image
import os.path
import glob
import cv2

def convertjpg(jpgfile, outdir):
    img = Image.open(jpgfile).convert("RGB")
    width_in, height_in, channel = np.array(img).shape
    dim_diff = np.abs(height_in - width_in)

    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = (0, 0, pad1, pad2) if height_in <= width_in else (pad1, pad2, 0, 0)
    top, bottom, left, right = pad

    pad_value = 0
    try:
        img_pad = cv2.copyMakeBorder(np.array(img), top, bottom, left, right, cv2.BORDER_CONSTANT, None, pad_value)
        dst_size = (256, 256)
        img_resize = cv2.resize(img_pad, dst_size, interpolation=cv2.INTER_AREA)
        img_resize = Image.fromarray(img_resize)
        img_resize.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

for jpgfile in glob.glob("img\\test_orgin\\A\\*.jpg"):
    if not os.path.exists("{}".format("TRAIN")):
        os.mkdir("{}".format("TRAIN"))
    convertjpg(jpgfile, "{}".format("TRAIN"))


