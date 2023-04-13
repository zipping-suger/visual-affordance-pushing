#!/usr/bin/env python

import time
import matplotlib.pyplot as plt
from camera import Camera

camera = Camera(use_filter=True)
time.sleep(1)  # Give camera some time to load data

while True:
    color_img, depth_img = camera.get_data()

    # Image show
    plt.subplot(211)
    plt.imshow(color_img)
    plt.subplot(212)
    plt.imshow(depth_img)
    plt.show()
