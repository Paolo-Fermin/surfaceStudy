# Neural Network Wake Image Study

This directory contains the files necessary to create a neural network that will take input of dTdz and d0 (change of temperature wrt depth and initial depth) and will produce a wake image for Uy. Future work must be done to incorporate Uz into the network as well, but for now, we are predicting just the one image. 

## Table of Contents
* [Setup environment](#setup-environment)
* [Obtaining training data](#obtaining-training-data)
* [Running training](#running-training)
* [Visualization](#visualization)
* [Future Work](#future-work)


## Setup environment

You will need a few Python packages before getting started. You can use the pip package manager to install these. If you don't have pip installed, run:

```
$ sudo apt-get install python-pip
```

Once pip is installed, run:

```
$ pip install <package> 
```

The packages you will need are: 
- pytorch
- torchsummary
- pandas
- numpy
- matplotlib
- visdom


## Obtaining training data

To create OpenFOAM training data, first you must specify what dTdz and d0 values you want to test. These can be specified near the beginning of the `parameter_variation.py` script within the lists `temps` and `depths`. After specifying your input variables, run `python parameter_variation.py`

Each case takes about 3-4 hours to finish execution. It will run faster if you specify more processors in the `num_procs` variable at the top of `parameter_variation.py`. The results of each OpenFOAM case can be found in the directory `openfoamruns/`. 

Once you have finished OpenFOAM, run `python get_data.py` This script will read the Uy and Uz arrays that we care about, and organize them in the `/data/` folder to be read by the neural network. They will be stored in folders named by case input variables, much like how they are stored in the `openfoamruns/` folder. You will have to manually sort the cases into `train_data/`, `val_data/`, and `test_data/` folders. Copy the whole directory, not just the .csv files, into these folders. The directory names are how the network will determine the values of the input variables. (Sorry I didn't write a script for this - I figured there were few enough cases that it would be simpler to sort them manually). 

## Running training

The main components necessary for network training are split between three python scripts. 

### 1. wake_model.py

This script defines the architecture of the network. Pretty straightforward. Read pytorch documentation online if you want to modify the architecture. An important thing to note is that this is a convolutional neural network, and this means that the network is very sensitive to the dimensions of input and output tensors. If you would like to change the dimensions of the output tensor, you must do so by changing the hyperparameters of the network, you cannot change the dimensions directly. Because of this, there are two models contained in this script, WakeModelFull and WakeModelCropped, which will produce 128x1024 and 128x512 images, respectively. 

### 2. wake_dataset.py

This script contains a class that is called by `train_network.py` to compile all the data to be fed into the network defined by `wake_model.py`. It also contains helper functions to transform the data to fit the network. Some of these include cropping the data to 128x1024 or 128x512 to agree with the size of the output tensor of the network.

### 3. train_network.py 

This script is the engine that actually runs the training. It loads the network model and then loads both training and validation datasets, called by their directory names within the data/ folder. Then it calls a visdom server to display loss graphs live. If you are getting strange visdom errors, see the VISUALIZATION section below. These errors shouldn't affect the training process itself. You can change the training parameters such as number of epochs, learning rate, optimizer, loss function, etc in this script. Once the script is done execution, it will store the network's weights in the `logs/` folder, where they can be accessed for later use.

To run the actual training, just call `python train_network.py [--epochs EPOCHS] [--crop]`

The optional parameter `epochs` allows you specify the number of epochs to train with. Default value is 1000, which is admittedly pretty low. The network takes about 15 min / 1000 epochs to train. The optional parameter `crop` allows you to specify whether to train the network on full images or half images. The reason for this is that the wake typically forms in only the first half of the image, and the second half is just background values. I am assuming that cutting the image in half would help the network focus on just the interesting parts of the wake image. 

## Visualization
During training, make sure to open up another terminal and call `visdom`

This will open up a visdom server that can be accessed by an internet browser on `localhost:8097`. Here, you can see the network's loss and validation loss graphs during training. 

After training, if you would like to see the actual predictions of the network, run:

```
$ python test_vis.py [--num MODEL NUM] [--crop]
```

The `--num` flag allows you to specify exactly which network's weights to load from the logs/ folder. By default, it will load the most recent one. The `--crop` flag is used when the network was trained with data that has been cropped to 128x512.

### Future Work 

1. **Two output images** The network as it stands is able to learn the Uy wake, or the Uz wake, but not both at the same time. I never really got around to implementing both. It should be possible by perhaps adding a second channel to the output tensor. 

2. **Low-pass filter** Some of the images that the network produces have weird noise in the background areas of the image. It should be simple to add a filter to floor all these stray noise values to the background value

3. **Cropping** The interesting part of the wake is mostly contained within just one half of the image. By removing the other half, hopefully the network will be able to more accurately predict the interesting parts of the image. I have implemented a model and some training options to account for this, but didn't really explore it fully

4. **Transposing the input image** While the actual values contained in the training wake images is accurate, they were not scaled nor transposed to accurately reflect the real world x and y coordinates. Firstly, the x values are way too small because I did not consider the velocity of the ship. Secondly, the y values in the original data csv (`/openfoamruns/_case_/postProcessing/sample/_timestep_/surface_U.xy`) are not included in the data that is fed into the network `Uy.csv`. Modifications should be made to either the `plotFarWake.m` octave script or the `get_data.py` python script to include the correct y values for consideration. This is important because convolutional networks learn spatial relationships, so accurate spatial dimensions may improve the predictive power of the network. 

5. **Timestep data in training** In `/openfoamruns/_case_/postProcessing/sample/` you will find wake images for every 30th timestep. This greatly increases the amount of training data available to the case, if only the network architecture were slightly modified to accept a third input, time. With this increase in input, the network would hopefully also be able to interpolate the wake image in between timesteps. 

