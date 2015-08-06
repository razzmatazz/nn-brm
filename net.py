import pickle
import random
import math
import time

from images import print_images

SIGMA = .01

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
    def __init__(self, filename, insz, hiddensz):
        self.filename = filename
        self.insz = insz
        self.hiddensz = hiddensz

        self.weights = [
            random.uniform(-.05, .05)
            for _ in xrange(insz)
            for _ in xrange(hiddensz)]

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
            for input in random.sample(input_list, 1):
                mod_inputs.append(input)
                mod_inputs.append(self.backwards(self.forward(input)))

            print_images(mod_inputs)
            self.save()

    def save(self):
        with open(self.filename, "w") as file:
            file.write(pickle.dumps(self))

    @staticmethod
    def load(filename):
        with file(filename) as f:
            net = pickle.loads(f.read())
            assert net.filename == filename

            return net

    def train_input(self, input):
        assert len(input) == self.insz

        hidden = self.forward(input)

        pos_gradient = [i * h
                        for i in input
                        for h in hidden]

        # compute negative gradient
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

