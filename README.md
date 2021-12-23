# CS-433 Machine Learning Project : Road Segmentation



## Project Members:

- Ahmed Nour Achich
- Mohamed Aziz Ben Chaabane
- Ahmed Amine Ghariani

## Description 

This project proposes a method for segmenting satellite images by road detection.
The classifier is consists of a form of Convolutional Neural Network called [UNet](https://en.wikipedia.org/wiki/U-Net), which outputs whether or not each pixel in an input image is considered to be part of a road.
The training set includes 100 satellite images of size 400x400 and their corresponding ground truth.
There are 50 satellite images in the testing set of size 608x608. 

## How to run

Install required packages with either ```pip install -r requirements.txt``` or ```conda install requirements.txt```.

Download the model from this [link](https://www.ab3athyawalid.com) , it needs to be in the root folder from where the python file is run.

To run the program call ```python3 run.py ``` from root directory to create submission with best pre-trained model.  

## File overview

- run.py : contains the code for running the project and generating a csv file submission.
- unet.ipynb : trained model
- requirements.txt : required packages list


 