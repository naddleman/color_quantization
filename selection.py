"""Selects k colors to best represent an image"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Constants
filename = "kodim23.png"
K = 6

im = Image.open(filename)
im = im.convert('HSV')
hue, sat, val = im.split()
harray = np.array(hue)
h, w = harray.shape
hvec = np.reshape(harray, -1)
hues = np.array(range(256))
color_ds= np.abs(hues - hues[:,None])
color_ds= np.where(color_ds<=128, color_ds, 256-color_ds)
def kmeans(vec, k):
    means = np.random.randint(0,256,k)
    convergence = [128]*k
    while sum(convergence) > 10:
        print(sum(convergence))
        d_to_mean = color_ds[vec][:,means]
        assignments = np.argmin(color_ds[vec][:,means], axis=1)
        for i in range(k):
            v = vec[assignments==i]
            v_mean = v.mean()
            convergence[i] = abs(means[i] - v_mean)
            means[i] = v_mean
        print(convergence)
    out = np.empty(hvec.shape)
    for i in range(k):
        out[assignments == i] = means[i]
    return out

hue_quantized = Image.fromarray(np.reshape(kmeans(hvec, K), harray.shape))
hue_quantized = hue_quantized.convert('L')

#quantize saturation (and boost it)
def boost(arr):
    return np.ceil(arr / 64) * 64

boosted_S = boost(np.array(sat))
boosted_V = boost(np.array(val))
s_out = Image.fromarray(boosted_S).convert('L')
v_out = Image.fromarray(boosted_V).convert('L')
out_img = Image.merge('HSV', [hue_quantized, s_out, v_out])

