# Alzheimer-brain-MRI-image-classification

This repository is dedicated to the research I conducted at the 2022 Advanced Academy for Research and Development summer program.  To run the file, you need a data folder named "data" with "train", "test", and "val" folders, each containing brain MRI images.  The CNN uses pretrained VGG16 layers connected to a batch normalization layer followed by alternating layers of 1000 neurons and 0.6 dropout.

## Abstract

Alzheimer’s disease is a neurologic disorder that hinders many elderly people from being able to live fulfilling lives. There is no cure for this disease, but patients can get medication to improve cognitive function. In order for patients to get more effective treatment, they need to be accurately diagnosed with the disease before it gets worse. In this research, a deep convolutional neural network was developed to predict the severity of early-stage Alzheimer’s disease based on brain MRI images. We compared several of the most commonly used pre-trained convolutional neural network architectures, such as VGG16, VGG19, InceptionV3, ResNet50, Xception, and DenseNet201. Our new finding is that VGG16 can make predictions with the highest accuracy.  The neural network has been fine-tuned by varying hyperparameters to maximize the performance of the model. By connecting the output of the VGG16 model to a batch normalization layer followed by four layers of 1000 neurons with a dropout rate of 0.6 between each layer, this model achieved an accuracy of 99.68% on the testing set. While other models can distinguish between no Alzheimer’s disease and severe Alzheimer’s disease, our model can differentiate the more subtle cases of no, very mild, and mild Alzheimer’s disease. Therefore, our approach may promptly and accurately diagnose the early stages of Alzheimer’s disease and help patients to get the necessary treatment before the noticeable symptoms appear.![image](https://user-images.githubusercontent.com/49625502/214154639-59934e18-2c99-4c4f-94e0-87a4b790242e.png)
