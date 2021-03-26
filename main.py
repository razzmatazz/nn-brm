#!/usr/bin/env python2

import sys
import math

from images import IMAGES, IMAGE_DIM, print_images, read_image
from net import Network

LAYER1_FILENAME = "_layer1.pkl"
HIDDEN_LAYER_SIZE = 64

def run():
    try:
        net = Network.load(LAYER1_FILENAME)
    except IOError:
        net = Network(LAYER1_FILENAME,
                      IMAGE_DIM * IMAGE_DIM,
                      HIDDEN_LAYER_SIZE)

    input_list = [list(read_image(ascii)) for ascii in IMAGES]

#    for _ in xrange(10):
    while True:
        net.train_input_list(input_list)

    pass

if __name__ == "__main__":
    run()
