# Keras implementation of "Wide Residual Networks"
This repo contains the code to run Wide Residual Networks using Keras.
- Paper (v1): http://arxiv.org/abs/1605.07146v1 (the authors have since published a v2 of the paper, which introduces slightly different preprocessing and improves the accuracy a little).
- Original code: https://github.com/szagoruyko/wide-residual-networks


## Dependencies:
- [numpy](https://github.com/numpy/numpy), [Keras](https://github.com/fchollet/keras) and it's dependencies (including the default [tensorflow](https://github.com/tensorflow/tensorflow) backend) can be installed with:
  - `sudo apt-get install python-pip python-dev gfortran libblas-dev liblapack-dev libhdf5-serial-dev libatlas-base-dev`
    - Note BLAS/LAPACK/ATLAS make linear algebra/numpy operations much much faster ([check numpy was installed against numpy](http://stackoverflow.com/a/19350234/5179470) with `import numpy as np; np.__config__.show()` ), and HDF5 dependencies allow saving/loading of trained models.
  - `sudo pip install -r requirements.txt` which includes TensorFlow backend (now the Keras default); alternatively install the Theano backend with `sudo pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps`  
- To plot the architecture of the model used (like the plot of the WRN-16-2 architecture plotted [below](#example-plot)), you need to install `pydot` and `graphviz`. I had to install the C-version for `graphviz` first (following [comments in this issue](https://github.com/Theano/Theano/issues/1801#issuecomment-64912809)):

```
$ sudo apt-get install graphviz
$ sudo pip install -I pydot==1.1.0
$ sudo pip install -I graphviz==0.5.2
```


## Training Details:
Run the default configuration (i.e. best configuration for CIFAR10 from original paper/code, WRN-28-10 without dropout) with:

```
$ python main.py
```

There are three configuration sections at the top of `main.py`:
- [DATA CONFIGURATION](https://github.com/asmith26/wide_resnets_keras/blob/master/main.py#L34-48): Containing data details.
- [NETWORK/TRAINING CONFIGURATION](https://github.com/asmith26/wide_resnets_keras/blob/master/main.py#L50-87): Includes the main parameters the authors experimented with.
- [OUTPUT CONFIGURATION](https://github.com/asmith26/wide_resnets_keras/blob/master/main.py#L89-97): Defines paths regarding where to save model/checkpoint weights and plots.


## Results and Trained models:
- ***WRN-40-4 no dropout***:
  - Using the same values in **main.py** except `depth=40` and `k:=widen-factor=4`, I obtained a **test loss = 0.37** and **accuracy = 0.93**. This test error (i.e. 1 - 0.93 = **7%**) is a little higher than the reported result (Table 4 states the same model obtains a test error of *4.97%*); see the note below for a likely explanation.
  - You can find the trained weights for this model at **models/WRN-40-4.h5**, whilst **[models/test.py](https://github.com/asmith26/wide_resnets_keras/blob/master/models/test.py)** provides an example of running these weights against the test set.
    - *WARNING:* These weights were obtained using the *Theano* backend - I am currently unable to reproduce these results using these trained weights with the TensorFlow backend. 

**Note:** I have not followed the exact same preprocessing and data augmentation steps used in the paper, in particular:

- "global *contrast* normalization", and
- "random crops from image padded by 4 pixels on each side, filling missing pixels with reflections of original image", which appears to be implemented in [this file](https://github.com/szagoruyko/wide-residual-networks/blob/8b166cc15fa8a598490ce0ae66365bf165dffb75/augmentation.lua).

Ideally, we will add such methods directly to the [Keras image preprocessing script](https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py).


##<a name="example-plot">WRN-16-2 Architecture</a>
![WRN-16-2 Architecture](models/WRN-16-2.png?raw=true "WRN-16-2 Architecture")
