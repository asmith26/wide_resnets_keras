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


## Results:
- Running **WRN-40-4 no dropout** now.

##<a name="example-plot">WRN-16-2 Architecture</a>
![WRN-16-2 Architecture](models/WRN-16-2.png?raw=true "WRN-16-2 Architecture")
