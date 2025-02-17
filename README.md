# Multi-label Clothing Classification Predicting Type and Color

## Table of Contents
- [Background](#background)
- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## Background

Matos Fashion is a rising startup in the e-commerce fashion world, currently facing challenges in managing and categorizing an ever-growing inventory of products. Their primary focus is on two types of clothing: T-shirts and Hoodies. To improve operational efficiency and simplify the shopping experience for customers, Matos Fashion aims to develop an automatic product classification system using product images. This system is designed not only to categorize products by type (such as T-shirts or Hoodies) but also by color (Red, Yellow, Blue, Black, White).

Multi-label classification is a machine learning task where a data instance can belong to multiple classes or labels simultaneously. Unlike traditional binary or multi-class classification, where an instance is assigned to only one class/category, multi-label classification allows multiple labels to be assigned to a single data instance. This is particularly useful for complex categorization tasks like the one Matos Fashion is addressing.

This project is part of the [Penyisihan Hology 7 Data Mining Competition 2024](https://www.kaggle.com/competitions/penyisihan-hology-7-data-mining-competition/overview).

## Problem Statement

The goal of this project is to build a deep learning model that can predict two labels for each clothing image: its type (T-shirt or Hoodie) and its color (Red, Yellow, Blue, Black, White). The model uses convolutional neural networks (CNN) and pre-trained architectures such as MobileNetV2 and DenseNet121 to achieve high performance and generalization capabilities.

## Approach

The approach involves two steps:
1. **Clothing Type Prediction**: The model first predicts whether the image is a **T-shirt** (Kaos) or a **Hoodie**. This task uses a MobileNetV2 model, which achieves an accuracy of **97%** on predicting the clothing type.
2. **Color Prediction**: Once the type is predicted, the model then predicts the color of the clothing using a DenseNet121 model. This second model achieved an accuracy of **0.95** on the color classification task.

After both predictions are made (type and color), the results are concatenated into a final submission dataset. This process ensures that each image has both a type and color label, fulfilling the multi-label classification requirement. The final result for the competition submission achieved an accuracy of **0.905982**.

## Key Features

- **Type Prediction:** Classifies images into two categories: T-shirt and Hoodie.
- **Color Prediction:** Classifies images into five categories: Red, Yellow, Blue, Black, and White.
- **Multi-label Classification:** Each image can be associated with both a clothing type and a color label simultaneously, enabling a more detailed classification process.
- **Transfer Learning:** Utilizes pre-trained models such as MobileNetV2 and DenseNet121 to leverage features learned from large-scale datasets like ImageNet.

## Model Architecture

To solve the problem, the project uses pre-trained models as a base and fine-tunes them for the specific task of clothing type and color prediction. Transfer learning is employed to benefit from existing knowledge embedded in models trained on vast image datasets.

The model architecture involves:
1. A base pre-trained model (e.g., MobileNetV2 or DenseNet121) for feature extraction.
2. Custom layers on top of the base model for classification tasks, which includes dense layers and softmax activation for multi-class outputs.
3. A final softmax layer that outputs the probability of each class (clothing type and color) for each image.

The model is trained with the objective to minimize the categorical cross-entropy loss, suitable for multi-class classification tasks.

## Dataset

The dataset consists of clothing product images categorized by type (T-shirt or Hoodie) and color (Red, Yellow, Blue, Black, White). The data was split into training, validation, and test sets, and preprocessing steps, such as rescaling and augmentation, were applied to ensure the model generalizes well to new data.

## Training & Evaluation

During training, the model uses data augmentation and validation steps to ensure robustness. The dataset was divided into 80% training, 10% validation, and 10% test, with a batch size of 32 images. After training, the model's performance is evaluated using accuracy metrics on the test set, with a high level of accuracy achieved in both clothing type and color prediction.

## Results

The model demonstrated strong performance in both clothing type and color classification tasks, achieving an accuracy of **97%** for type prediction using MobileNetV2, and **95%** for color prediction using DenseNet121. The final submission, which combines both results (type and color predictions), achieved an accuracy of **0.905982** in the competition.

## Future Work

- Further fine-tuning the pre-trained models and exploring more advanced architectures.
- Incorporating more data augmentation techniques to improve robustness.
- Deployment of the model for real-time clothing recognition in e-commerce platforms.

## Contributors

- **Steve Marcello Liem**
- **Davin Edbert Santoso Halim**
- **Dave Christian Thio**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
