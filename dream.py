#!/usr/bin/env python

import sys
import time

from main import LAYER1_FILENAME
from images import IMAGE_DIM, print_images
from net import Network

def run():
    net = Network.load(LAYER1_FILENAME)

    input = [.0] * IMAGE_DIM * IMAGE_DIM

    while True:
        hidden = net.forward(input)
        input = net.backwards(hidden)
        print_images([input])

        time.sleep(.075)

if __name__ == "__main__":
    run()
