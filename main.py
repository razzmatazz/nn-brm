#!/usr/bin/env python

import sys
import math
import random
import time

from images import IMAGES, IMAGE_DIM

NUM_WEIGHT_FACTOR = 3
SIGMA = .01

def rounded_v(v, s):
    return [round(i, s) for i in v]



def stochastic(v):
    return 1 / (1 + math.exp(-v))

def sto(v):
    return 1 if random.random() < stochastic(v) else 0


LAST_TIME = {"shown": None}

def can_show_something():
    if not LAST_TIME["shown"] or (time.time() - LAST_TIME["shown"] > 0.5):
        LAST_TIME["shown"] = time.time()

        return True
    else:
        return False

class Network(object):
    def __init__(self, insz, hiddensz):
        self.insz = insz
        self.hiddensz = hiddensz

        self.weights = [
            random.uniform(-.05, .05)
            for _ in xrange(insz)
            for _ in xrange(hiddensz)]

        #print "initial weights", self.weights

    def train_input_list(self, input_list):
        assert len(input_list) > 0
        assert all(len(input) == self.insz for input in input_list)

        weight_deltas = [.0] * len(self.weights)

        sample_size = len(input_list) / 3
        for input in random.sample(input_list, sample_size):
            train_deltas = self.train_input(input)

            weight_deltas = [a + b for a, b in zip(weight_deltas, train_deltas)]

        self.apply_deltas(weight_deltas)

        mod_inputs = []

        if can_show_something():
            for input in input_list:
                mod_inputs.append(input)
                mod_inputs.append(self.backwards(self.forward(input)))

            print_images(mod_inputs)

    def train_input(self, input):
        assert len(input) == self.insz

#        print "---"
#        print "weights", self.weights
#        print "input", input

        hidden = self.forward(input)
#        print "hidden", hidden

        pos_gradient = [i * h
                        for i in input
                        for h in hidden]


        # compute negative gradien
        input_mod = self.backwards(hidden)
        hidden_mod = self.forward(input_mod)

        neg_gradient = [i * h
                        for i in input_mod
                        for h in hidden_mod]

        return [SIGMA * (p - n)
                for p, n in zip(pos_gradient, neg_gradient)]

    def forward(self, input):
        assert len(input) == self.insz

        return [sto(sum(input[ix] * self.weights[ix * self.hiddensz + ox]
                        for ix in range(self.insz)
                   ))
                for ox in range(self.hiddensz)]

    def backwards(self, hidden):
        assert len(hidden) == self.hiddensz

        return [sto(sum(hidden[ox] * self.weights[ix * self.hiddensz + ox]
                    for ox in range(self.hiddensz)
                   ))
                for ix in range(self.insz)]
       
    def apply_deltas(self, deltas):
        assert len(deltas) == len(self.weights)

        self.weights = [w + delta
                        for w, delta in zip(self.weights, deltas)]


def num_to_char(n):
    return (" " if n < .2
            else ("." if n < 0.4
                  else ("+" if n < 0.6
                        else ("x" if n < 0.8
                              else "#"))))


def print_images(imgs):
    print ("-" * IMAGE_DIM + " ") * len(imgs)
    for line in range(IMAGE_DIM):
        for img in imgs:
            print "".join(map(num_to_char, img[line * IMAGE_DIM:][:IMAGE_DIM])),

        print ""
            

def read_image(ascii):
    for a in ascii:
        if a == " ":
            yield .0
        elif a == ".":
            yield .3
        elif a == "#":
            yield 1
        else:
            raise Exception("invalid image char '" + a + "'")

def run():
    net = Network(5 * 5, 16)

    input_list = [list(read_image(ascii)) for ascii in IMAGES]

#    for _ in xrange(10):
    while True:
        net.train_input_list(input_list)

    pass

if __name__ == "__main__":
    run()
