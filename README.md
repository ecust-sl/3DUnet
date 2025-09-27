# TabNet

## Requirements
Our code works with the following environment.
- `torch==2.1.2`
- `torchvision==0.6.2`
- `opencv-python==4.4.0.42`

## Datasets
We use datasets in our paper.
First you need to follow the steps in step.txt in the nnUnet file directory to preprocess the image.
You need to put the dataset in the Task504_All folder and meet the following directory structure
- Task504_All
  - imagesTr
  - imagesTs
  - labelsTr
  - labelsTs

## Running

Out model training：python train.py
Out model testing: python inference.py

## Tip
1.Two or more categories will require you to adjust the category code
2.For training and testing, you simply need to modify the corresponding paths.
3.The specific network architecture is located within the unet3d folder.

