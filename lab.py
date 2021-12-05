"""Clustering and analysis of images' colors in L*a*b color space"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image, ImageCms
from skimage import io, color

# Constants
buckets = 8
args = sys.argv
if len(args) != 2:
    sys.exit("Usage: python lab.py <filename>")

filename = args[1]
#out_file = f"{filename[:-4]}_{K}_clusters.png"

im = io.imread(filename)
Lab = color.rgb2lab(im)
a = Lab[:,:,1]
avec = np.reshape(a, -1)
b = Lab[:,:,2]
bvec = np.reshape(b, -1)
sample = np.random.choice(len(avec), 1000)
plt.scatter(avec[sample], bvec[sample], marker='.')
plt.xlabel('a')
plt.ylabel('b')
plt.show()

# Let's do something *like* median cut
for i in range(buckets)

#harray = np.array(hue)
#h, w = harray.shape
#hvec = np.reshape(harray, -1)
#hues = np.array(range(256))
#color_ds= np.abs(hues - hues[:,None])
#color_ds= np.where(color_ds<=128, color_ds, 256-color_ds)
#def kmeans(vec, k):
#    means = np.random.randint(0,256,k)
#    convergence = [128]*k
#    while sum(convergence) > 10:
#        print(f"means: {means}")
#        d_to_mean = color_ds[vec][:,means]
#        assignments = np.argmin(color_ds[vec][:,means], axis=1)
#        for i in range(k):
#            v = vec[assignments==i]
#            v_mean = v.mean()
#            convergence[i] = abs(means[i] - v_mean)
#            means[i] = v_mean
#    out = np.empty(hvec.shape)
#    for i in range(k):
#        out[assignments == i] = means[i]
#    return out
#
#hue_quantized = Image.fromarray(np.reshape(kmeans(hvec, K), harray.shape))
#hue_quantized = hue_quantized.convert('L')
#
##quantize saturation (and boost it)
#def boost(arr):
#    return np.ceil(arr / 64) * 64
#
#boosted_S = boost(np.array(sat))
#boosted_V = boost(np.array(val))
#s_out = Image.fromarray(boosted_S).convert('L')
#v_out = Image.fromarray(boosted_V).convert('L')
#out_img = Image.merge('HSV', [hue_quantized, s_out, v_out])
#out_img = out_img.convert('RGB')
#print(f"saving {out_file}")
#out_img.save(out_file, 'PNG')
#
