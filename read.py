#!/usr/bin/env python2

import sys

from main import read_image, LAYER1_FILENAME
from images import IMAGE_DIM, print_images
from net import Network

def read():
    print "enter image, %d lines:" % (IMAGE_DIM,)

    if len(sys.argv) == 2:
        image_f = file(sys.argv[0])
    else:
        image_f = sys.stdin

    image = ""
    for num in range(IMAGE_DIM):
        line = image_f.readline().strip("\n")

        if len(line) > IMAGE_DIM:
            line = line[:IMAGE_DIM]
        else:
            line = line + " " * (IMAGE_DIM - len(line))

        if len(line) != IMAGE_DIM:
            print "unexpected len of line %d: %d (%d chars expected)" % (
                num, len(line), IMAGE_DIM)
            sys.exit(1)
            
        image += line

    assert len(image) == IMAGE_DIM * IMAGE_DIM

    return image

def run():
    image = read()
    input = list(read_image(image))

    net = Network.load(LAYER1_FILENAME)

    hidden = net.forward(input)
    for _ in xrange(5):
        print_images([net.backwards(hidden)])

if __name__ == "__main__":
    run()
