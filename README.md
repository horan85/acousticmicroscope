

# Deep Learning-Based Super-Resolution Applied to Acoustic Microscopy

**A github repository for enhancing acoustic microscopy images**

In this repository You can find our code which impelments a superroslution neural network.

### Prerequisites-Installing

To run our code You need to install [Python](https://www.python.org/)  (*v3.5*) and  [Tensorflow](https://www.tensorflow.org/) (v1.12.0) and that is all.

### Running our code
 Our training script was implemented as a single file, all You have to do to train your models on your own data is to change the data loading part (marked as *#load the data from numpy*) and run the script.
 The implementation of the U-NET based network can be found in  [unet_train.py](https://github.com/horan85/acousticmicroscope/blob/master/unet_train.py)
 and the infernce based on a previously trained network can be found in  [unet_test.py](https://github.com/horan85/acousticmicroscope/blob/master/unet_test.py).

The training script contains running time data augmentation on the samples and this way it was able to generate a reasonably well working network using only a limited amount of data.

## Example
An example image showing the input and output image is displayed for qulaitative evaluation:

## Authors
**Akos Makra
Wolfgang Bost
Marc Fournelle
Imre Kallo
Andras Horvath
Miklos Gyongy** 
