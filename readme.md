# About

This is a toy implementation of a [boltzman restricted machine](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine).


You can also add more "images" in `images.py` to train the network.

## Usage

Run ./main.py for a while so network learns from `images.py`.

Then you can run `read.py` to evaluate an image agains network, or
or `dream.py` and you can see how the network fluctuates around
it's "fantasies".

You can also do `cat img-2.img > ./read.py` to check how network
reflects on the image supplied.
