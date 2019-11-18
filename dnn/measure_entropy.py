import argparse
import glob
import numpy as np

from skimage import io, color
from skimage.measure import shannon_entropy

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, required=True)
args = parser.parse_args()

images = sorted(glob.glob('{}/*.png'.format(args.image_dir)))

entropy_values = []
for image in images:
    rgbimg = io.imread(image)
    grayimg = color.rgb2gray(rgbimg)
    entropy_value = shannon_entropy(grayimg)
    entropy_values.append(entropy_value)
    print(entropy_value)
print(np.average(entropy_values))
