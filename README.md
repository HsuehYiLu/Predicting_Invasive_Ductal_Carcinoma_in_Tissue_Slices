# Predicting_Invasive_Ductal_Carcinoma_in_Tissue_Slices
Predicting Invasive Ductal Carcinoma in Tissue Slices using two pre-trained convolutional neural networks to detect IDC in the tissue slice images

# Final Project Report
## Predicting Invasive Ductal Carcinoma in Tissue Slices
*Yue Li, Hsueh-Yi Lu*
*Advisor: Amir Jafari*
*Fall 2022*
*Dec 13, 2022*

## Table of Contents
1. [Introduction](#introduction)
2. [Description of Dataset](#description-of-dataset)
3. [Description of the Model and Algorithm](#description-of-the-model-and-algorithm)
   - Convolutional Neural Network
   - Transfer Learning
   - Pre-trained Model - VGG
   - Pre-trained Model - ResNet
   - Cyclical Learning Rates (CLR)
   - General Adversarial Network (GAN)
4. [Experimental Setup](#experimental-setup)
   - Data Preprocessing
   - Exploratory Data Analysis (EDA)
   - Train, Validation, and Test Split
   - Data Augmentation
   - Metric Selection
   - Model Selection
   - Learning Rate Search
   - Train Processing
5. [Results & Summary](#results--summary)
6. [Future Work](#future-work)
7. [References](#references)

## Introduction
Invasive ductal carcinoma (IDC) is one of the most common types of breast cancer. Detecting IDC in tissue slice images is currently a manual and time-consuming process performed by pathologists. Deep learning techniques, such as Convolutional Neural Networks (CNNs), can potentially aid in automating this process and speed up diagnosis.

## Description of Dataset
The dataset used in this project is sourced from Kaggle and contains 162 whole mount slide images of Breast Cancer (BCa) specimens scanned at 40x. From these slides, 277,524 patches of size 50x50 were extracted, consisting of 198,738 IDC-negative patches and 78,786 IDC-positive patches.

## Description of the Model and Algorithm
- Convolutional Neural Network (CNN): Two pre-trained CNN models, VGG16 and ResNet18, were used for the classification task. CNNs are deep learning algorithms that assign importance to various aspects of an image to differentiate between classes.
- Transfer Learning: Pre-trained models from ImageNet were used to leverage their learned features and adapt them to the IDC classification task.
- Cyclical Learning Rates (CLR): A learning rate search method was used during the training process to optimize the learning rate between specified boundaries, resulting in improved convergence and stability.
- General Adversarial Network (GAN): A cGAN (conditional GAN) was implemented for data augmentation and color normalization to increase the diversity of images and improve model performance.

## Experimental Setup
Data preprocessing, exploratory data analysis, train-validation-test split, data augmentation, and metric selection were performed to prepare the data and set up the evaluation process.

## Results & Summary
ResNet18 outperformed VGG16 in terms of accuracy and training time. The use of Cyclical Learning Rates (CLR) improved the stability and convergence speed of both models.

## Future Work
Potential future work includes generating color-normalized images using Pix2Pix network and incorporating them into the training set to further improve model performance.

## References
The report includes a list of references used for the project, covering various sources related to convolutional neural networks, datasets, transfer learning, and GANs.

1. [Convolutional neural networks: an overview and application in radiology](https://link.springer.com/article/10.1007/s13244-018-0639-9) | Insights into Imaging
2. [Breast Cancer Notebook: Breast Cancer](https://www.kaggle.com/paultimothymooney/breast-histopathology-images) | Kaggle
3. [Breast Histopathology Images](https://www.kaggle.com/paultimothymooney/breast-histopathology-images) | Kaggle
4. [ResNet and VGG: Why is resnet faster than vgg](https://stats.stackexchange.com/questions/364882/why-is-resnet-faster-than-vgg) | Cross Validated
5. [Cyclic Learning rate: Super Convergence with Cyclical Learning Rates in TensorFlow](https://towardsdatascience.com/super-convergence-with-cyclical-learning-rates-in-tensorflow-d8a245fd8a4) | by Christopher Thomas BSc Hons. MIAP | Towards Data Science
6. Ibraham, Kandel., Mauro, Castelli., (2020). [The effect of batch size on the generalizability of the convolutional neural networks on a histopathology dataset](https://www.sciencedirect.com/science/article/pii/S2213158220302296). Volume 6, Issue 4, Science Direct.
7. Cruz-Roa, A., Basavanhally, A., González, F., Gilmore, H., Feldman, M., Ganesan, S., Shih, N., Tomaszewski, J., & Madabhushi, A. (2014). [Automatic detection of invasive ductal carcinoma in whole slide images with convolutional neural networks](https://doi.org/10.1117/12.2043872). SPIE Proceedings.
8. [Pix2Pix Kaggle: Stain Transfer w/ Pix2Pix; PyTorch/Lightning](https://www.kaggle.com/mrgravelord/colored-and-stained-histology-mnist) | Kaggle
9. Salehi, P., & Chalechale, A. (2020). [Pix2Pix-based stain-to-stain translation: A solution for robust stain normalization in histopathology images analysis](https://doi.org/10.1109/mvip49855.2020.911689). 2020 International Conference on Machine Vision and Image Processing (MVIP).




## Getting Started

 To get started, you will need to install the following dependencies:
 
```
 Python 3.6+
 PyTorch# Torchvision
 NumPy
 Scikit-learn
```

# Once you have installed the dependencies, you can clone the repository and run the following command to install the project dependencies:

```commandline
  pip install -r requirements.txt
```

## Usage

 To train the models, you can run the following command:
```commandline
 python train.py
```

 This will train the models and save the trained models to the `models` directory.

 To evaluate the models, you can run the following command:

```commandline
  python evaluate.py
```

 This will evaluate the models on the test set and print the accuracy.


