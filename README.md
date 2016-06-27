# Keras (Theano) implementation of "Wide Residual Networks"
This repo contains the code to run Wide Residual Networks using Keras running Theano.
- Paper: http://arxiv.org/abs/1605.07146
- Original code: https://github.com/szagoruyko/wide-residual-networks


## Dependencies:
- [Keras](https://github.com/fchollet/keras), [numpy](https://github.com/numpy/numpy) - can be installed with `sudo pip install -r requirements.txt`.
- [Theano](https://github.com/Theano/Theano); install latest version with `sudo pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps`.
- To plot the architecture of the model used (like the plot of the WRN-16-2 architecture plotted [below](#example-plot)), you need to install `pydot` and `graphviz`. I had to install the C-version for `graphviz` first (following [comments in this issue](https://github.com/Theano/Theano/issues/1801#issuecomment-64912809)):

```
$ sudo apt-get install graphviz
$ sudo pip install pydot
$ sudo pip install graphviz
```


## Usage Details:
Run the default configuration (i.e. best configuration for CIFAR10 from original paper/code, WRN-28-10 without dropout) with:

```
$ python main.py
```

There are three configuration sections at the top of `main.py`:
- [NETWORK CONFIGURATION](https://github.com/asmith26/wide_resnets_keras/blob/master/main.py#L32-59): Includes the main parameters the authors experimented with.
- [DATA CONFIGURATION](https://github.com/asmith26/wide_resnets_keras/blob/master/main.py#L61-75): Containing data details.
- [OUTPUT CONFIGURATION](https://github.com/asmith26/wide_resnets_keras/blob/master/main.py#L77-85): Defines paths regarding where to save model/checkpoint weights and plots.


## Results and Trained models:
- ***WRN-40-4 no dropout***:
  - Using the same values in **main.py** except `depth=40` and `widen-factor=4`, I obtained a **validation loss = 0.37** and **accuracy = 0.93**. This test error (i.e. 1 - 0.93 = **7%**) is a little higher than the reported result (Table 4 states the same model obtains a test error of *4.17%*); see the note below for a likely explanation.
  - You can find the trained weights for this model at **models/WRN-40-4.h5**, whilst **models/test.py** provides an example running these weights against the test set.

**Note:** I have not followed the exact same preprocessing and data augmentation steps used in the paper, in particular:

- "global *contrast* normalization", and
- "random crops from image padded by 4 pixels on each side, filling missing pixels with reflections of original image", which appears to be implemented in [this file](https://github.com/szagoruyko/wide-residual-networks/blob/master/augmentation.lua).

Ideally, we will add such methods directly to the [Keras image preprocessing script](https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py).


##<a name="example-plot">WRN-16-2 Architecture</a>
![WRN-16-2 Architecture](models/WRN-16-2.png?raw=true "WRN-16-2 Architecture")
