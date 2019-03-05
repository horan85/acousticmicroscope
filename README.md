

# Deep Learning-Based Super-Resolution Applied to Acoustic Microscopy

**A github repository for acoustic microscopy image resolution enhancement**

In this repository You can find our code which implements a superresolution neural network.

### Prerequisites-Installing

To run our code You need to install [Python](https://www.python.org/)  (*v3.5*) and  [Tensorflow](https://www.tensorflow.org/) (v1.12.0) and that is all.

### Running our code
 Our training script was implemented as a single file, all You have to do to train your models on your own data is to change the data loading part (marked as *#load the data from numpy*) and run the script.
 The implementation of the U-NET based network can be found in  [unet_train.py](https://github.com/horan85/acousticmicroscope/blob/master/unet_train.py)
 and the inference based on a previously trained network can be found in  [unet_test.py](https://github.com/horan85/acousticmicroscope/blob/master/unet_test.py).

The training script contains running time data augmentation on the samples and this way it is able to train a network using only a limited amount of data, while providing promising results on the test set.

## Example
An example image showing the input and output image is displayed for qualitative evaluation:
![alt text](https://github.com/horan85/acousticmicroscope/raw/master/samples.png)
## Authors
**Ákos Makra
Wolfgang Bost
Imre Kalló
András Horváth
Marc Fournelle
Miklós Gyöngy** 
