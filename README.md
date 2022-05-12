# Applying Convolutional Neural Networks For Tumor Malignancy Classification

## Introduction and Objectives

Cancer is the second leading cause of death globally, taking the lives of over 8 million people every year. Tumor malignancy testing is a major part of cancer diagnosis since the treatment plans for benign and malignant tumors are vastly different. Therefore, high precision and high recall of tumor diagnosis are crucial for identifying proper treatment. In this project, we will be applying deep convolutional neural networks with different architectures to learn tumor representations and use them for classifying the tumor as either benign or malignant. We will explore pre-trained neural networks and optimize for multiple hyperparameters including regularization and learning rate. Precision and recall for each hyperparameter will be compared and the best model will be suggested.

## Methods

We plan to use the PatchCamelyon (PCam) dataset[1]. PCam consists of 327.680 color images (96x96px) extracted from histopathologic scans of lymph node sections. Each image is annotated with a binary label indicating the presence of metastatic tissue. The dataset is divided into a training set of 262.144 examples, and a validation and test set of both 32.768 examples, (218, 215, 215, respectively). A positive label indicates that the center 32x32px region of a patch contains at least one pixel of cancerous tumor tissue.

## Intended Experiments

As our primary evaluation metric, we will use the area under the receiver operating characteristic curve (AUROC). After visualizing the data and gathering basic statistics, we will properly perform data preprocessing. The initial plan is to start with a relatively simple CNNbased model, which will hopefully shed some light on potential problems and act as a setup for our final model. Then, we will be using pre-trained CNN-based models and try to customize or incrementally add complexity to those models to get some performance boost. As soon as our models become complex enough to overfit the datasets, we will introduce regularization and augment data if the need arises.

Finally, we will spend most of our time tuning the hyperparameters until we achieve low validation loss.
