"""visualizing decision trees by predicting pixel colors"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageCms
from skimage import io, color
from sklearn.tree import DecisionTreeRegressor

# Constants
filename = 'test2.png'
depths = [5,8,11,14,17,20]


im = io.imread(filename)
h,w = im.shape[0], im.shape[1]
r,g,b = [im[:,:,i] for i in range(3)]
rgb_vectors = [x.reshape(-1) for x in [r,g,b]]
pixels = []
for depth in depths:
    trees = [DecisionTreeRegressor(random_state=1001, max_depth=depth)
                        for _ in [r,g,b]]
    x_coords = np.arange(len(rgb_vectors[0])) % w
    y_coords = np.arange(len(rgb_vectors[0])) // w
    # create a vector of pixel coordinates 
    features = np.array([x_coords,y_coords]).T
    [tree.fit(features, rgb_vectors[i]) for i,tree in enumerate(trees)]
    predicted_pixels = [tree.predict(features).reshape(h,w) for tree in trees]
    pixels.append(predicted_pixels)

pixels = np.array(pixels).astype('uint8')
out = []
for i in range(len(depths)):
    color_channels = [[],[],[]]
    for c in range(3):
        im = Image.fromarray(pixels[i,c,:,:])
        color_channels[c] = im
    out.append(Image.merge('RGB', color_channels))
    

out[0].save('colortest.gif',save_all=True,append_images=out[1:],
                optimize=False,duration=250,loop=0)
