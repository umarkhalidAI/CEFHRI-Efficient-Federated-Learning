# CEFHRI-Efficient-Federated-Learning
Official implementation of the IROS Paper: CEFHRI: A Communication Efficient Federated Learning Framework for Recognizing Industrial Human-Robot Interaction

Arxiv Version https://arxiv.org/abs/2308.14965v1

# Environemnt
The code has been tested for Python 3.9+ and Cuda 11.7
# Dataset Preparation
## Dataset Location
We have posted the annotations of the datasets used in the paper.
The HRI_anno, coin_anno and inhard_anno have the train.txt and val.txt files where you can find the location of the videos relative to main_use.py
For Example: ./../HRI/videos/DeliverObject/v_DeliverObject_g05_c01.avi 0 indicates the relative location of a sample from the HRI30 dataset placed in the folder HRI/videos/... and has a label 0
## HRI30 Dataset
Download from https://zenodo.org/record/5833411
## COIN
Download from https://coin-dataset.github.io/
## InHard Dataset
Use the instructions at https://github.com/vhavard/InHARD
# Code Running
Example scripts can be found in the scripts folder where we have posted the bash files for three different datasets used in the paper.

