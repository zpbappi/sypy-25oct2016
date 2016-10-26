# sypy-25oct2016
Presentation and demo materials presented on 25th October 2016 in Sydney Python User Group (SyPy)

## mnist.py
Downloads the MNIST dataset and trains a network with 15 hidden layes with dense connection on the samples. At the end, outputs the accuracy of test dataset.

## mnist-analyze.py
Downloads the MNIST dataset and trains a network with no hidden layers on the samples. At the end, outputs the accuracy of test dataset. Then it asks to enter a digit (0-9) to visualize the learned weights. Enter any other number (< 0 or > 9) to exit.

## inception.py
Downloads the pre-trained InceptionV3 network, adds two extra layers on top of it, freezes the existing inceptionv3 layers so that we don't end up training it, then simply trains the last two added layers on the sample images provided. At the end, predicts whether it is an image of a dog or a cat from the test dataset. _Note:_ modfy the path in line 11.

## dogcat.zip
Includes the training and test images for training in inception.py

## ML-SyPy.pptx
Presentation slides.