# Applying Convolutional Neural Networks For Tumor Malignancy Classification

## Introduction and Objectives

Cancer is the second leading cause of death globally, taking the lives of over 8 million people every year. Tumor malignancy testing is a major part of cancer diagnosis since the treatment plans for benign and malignant tumors are vastly different. Therefore, high precision and high recall of tumor diagnosis are crucial for identifying proper treatment. In this project, we will be applying deep convolutional neural networks with different architectures to learn tumor representations and use them for classifying the tumor as either benign or malignant. We will explore pre-trained neural networks and optimize for multiple hyperparameters including regularization and learning rate. Precision and recall for each hyperparameter will be compared and the best model will be suggested.

## Methods

In this project, we utilized and compared convolutional and transformer deep learning architectures for lymph node cancer detection(1). To train the neural networks, we used PatchCamelyon (PCam) dataset(2). The original PCam consists of 327.680 color images (96x96px) extracted from histopathologic scans of lymph node sections. Each image is annotated with a binary label indicating the presence of metastatic tissue. Positive label indicates that the center 32x32px region contains malignant tissue whereas the negative label means that the central 32x32 patch doesn’t contain a tumor. Due to the large size of the original dataset, we decided to use a small fraction of the data points. In the initial round of training, we used 6763 images sampled from PCam, with 90% of the dataset reserved for training, 5% for validation and 5% for testing. In the subsequent rounds of training, we increased the dataset size to 10000 and used 80% for training, 10% for validation, and 5% for testing.
For the training data, we utilized data preprocessing to reduce computational complexity. We re-scaled the numeric values of pixels in all datasets to lie between 0 and 1. Pixel value rescaling was done to reduce computation complexity for both forward and back propagation and reduce the risk of overflow. As for data augmentation, we applied random rotation between 0o-45o , horizontal, vertical flips, shifted image along X and Y axis by 30%, and finally, used zoom-in and zoom-out methods. The main reason to do these kind of data augmentations is to make our training dataset as rich as possible, in other words, to make the dataset more representative.

We then proceeded to train multiple neural networks. First, we trained a custom network with multiple convolutional layers as the baseline. This model doesn’t use any pre-trained network and has 2.1 million trainable parameters. The architecture of the baseline network consists of 4 repeating blocks depicted in Figure 1. First comes the convolutional layer which applies multiple convolutional filters to create a feature map that reveals whether the learned features are present in the input. Then comes a maximum pooling layer that down samples the feature map. This has an effect of making the feature map less sensitive to changes in position of the detected feature and introduces local transition invariance. Next comes the dropout layer which is used to perform model averaging and regularization by randomly dropping its constituent units and their connections. Finally, there is the batch normalization layer which performs re- centering and re-scaling of its inputs. This makes the subsequent computations faster and more numerically stable by reducing the chance of overflow. The output of the 4 CNN blocks is flattened to a single vector and then interpreted by the multilayered perceptron with 4 dense layers depicted in Figure 3. MLP accepts the flattened feature vector learned by the preceding CNN blocks and then makes the decision on whether the input image contains malignant tumor in the central patch.

Apart from the custom CNN baseline, we utilized transfer learning and constructed 3 more networks which utilize VGG16 (14.8 million parameters), VGG19 (20.0 million parameters), InceptionV3 (21.8 million parameters) and ResNet-50 (23.5 million parameters) as base models initialized with weights from the ImageNet database(8). All layers of the pre-trained models were frozen. All 3 networks based on the pre-trained models take the output of the pre-trained models (excluding their fully connected layers), flatten it to a vector and feed it to the MLP in Figure 2. We chose to use the same MLP upper prediction layer across all tested architectures to make them more comparable. This way, we can directly relate the differences in the overall accuracy to the network’s ability to extract useful features from images.

In addition to the custom and pre-trained CNN models, we decided to utilize the transformer encoder-decoder architecture for malignancy classification. Transformer architecture(10) introduced in 2017 revolutionized the NLP field and significantly improved the state-of-the-art NLP performance thanks to the utilized multi-head attention mechanism which relates different positions of a single sequence to compute its representation. Attention takes query (Q ∈ 𝑅𝑅𝑑𝑑𝑘𝑘), key (K ∈ 𝑅𝑅𝑑𝑑𝑘𝑘), and value (V ∈ 𝑅𝑅𝑑𝑑𝑣𝑣) vectors and computes a weighted sum of values. Weight assigned to each value is computed by the compatibility function of query with corresponding key:

(1) Scaled dot-product attention function

$$ Attention(Q, K, V) = softmax({QK^T\over \sqrt{d_k}}) $$

The scaling factor $\sqrt{d_k}$ is introduced to increase numeric stability and avoid vanishing gradients at high dimensions of Q, or K
In multi-head attention with h heads, scaled dot product attention is applied in parallel to h different learned projections of query, key, and value vectors. h attention outputs for each of these projections are then concatenated and projected back the original dimension:

(2) Multi-head attention function

$$ MultiHead(Q,K,V) = Concat(head_1, ... , head_h) W^0 $$

(3) Equation of each head in multi-head attention

$$ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V) $$

Here weight parameters 𝑊𝑊𝑖𝑖 cast the Q, K, V vectors to a reduced dimension, producing h different projections. Meanwhile, 𝑊𝑊𝑂𝑂 introduces non-linearity and additional learnable parameters to produce an optimized overall attention value:

$$ W_i^Q \in {R^{d_{model} \cdot {d_{model} \over h}}}, W_i^K \in {R^{d_{model} \cdot {d_{model} \over h}}}, W_i^V \in {R^{d_{model} \cdot {d_{model} \over h}}}, W_i^O \in {R^{d_{model} \cdot d_{model}}} $$

Multi-head attention allows the transformer to attend to multiple regions of the input sequence at once and thus compute a more complete learned representation of the input. It is the basis of the transformer. On a higher level, transformer consists of 2 key blocks: transformer encoder, and transformer decoder. Transformer encoder produces a representation of the input sequence by applying multi-head self-attention. Meanwhile, the decoder generates the result by first applying the multi-head self-attention to the output of the previous decoder layer, and then applying encoder-decoder attention which uses encoder final state output as keys and values, and the output of the previous decoder as queries. Multiple layers of encoder and decoder may be stacked together to learn deeper patterns from the input data. However, despite its merits, transformers are typically applied to sequential data, such as text, and not images. To make images compatible with the transformer architecture, we patched the 32*32 cropped image into 196 patches, 5*5 pixels each. This procedure is demonstrated below:

![Picture1](https://user-images.githubusercontent.com/63436458/184296046-595afba2-80ee-430e-8bdb-c429b968f445.jpg)

*Figure 1. Image pre-processing for transformer encoder-decoder architecture. The image is first center-cropped to produce 32x32 fragment. The fragment is then patched into 5 x 5 pixel segments and the segments are reshaped into a sequence. Here we use an image from the utilized PCam dataset*

We then reshaped the patches into a sequential vector with 196 patches and used it as input for a simple 1-layer encoder decoder transformer illustrated below. Our transformer model has just 1 encoder and decoder layer and 15 million trainable parameters.

![Picture2](https://user-images.githubusercontent.com/63436458/184296099-3c5ae8cc-473b-4af6-9ff8-303d178c5a1b.png)

*Figure 2. Transformer encoder-decoder for malignancy classification. The input image is patched and reshaped into a vector, then fed into the encoder. The output of the encoder is then fed into the decoder’s cross-attention. Finally, the result is pooled and fed into the softmax classifier*

The input image is center cropped and then patched and reshaped into a vector, which is subsequently fed into the encoder. The encoder consists of 2 key sub-layers. The first sub-layer computes multi-head self-attention based on the sequence of image patches. The second sub-layer is a fully connected feed-forward neural network with 3 dense layers. The residual connection with subsequent normalization is used around each of the 2 encoder layers. The output of the encoder contains the hidden state for each patch in the sequence. The second major element of the transformer is the decoder layer, which consists of 3 key sub-layers. The first sub-layer computes multi-head self-attention based on the sequence of image patches. The second sub-layer computes multi-head encoder-decoder cross-attention based on the self-attention from the previous layer and the encoder output. The third sub-layer is a feed-forward neural network with 3 dense layers. The residual connection with subsequent normalization is used around each of the 3 layers. Finally, the normalized result is pooled to extract the key features and reduce dimensionality. The result of global average pooling is fed into the softmax classifier which predicts the probability of tumor malignancy.

Finally, we also developed a hybrid architecture which combines the benefits of convolutional neural networks for extracting useful features for the data and the benefits of transformer which can learn powerful representations of the input sequence thanks to the multi-head attention mechanism. To do this, we first center-cropped the image and then fed the output into VGG19. We then extracted the output of the fourth convolutional layer of VGG19 which contained 128 feature maps, each 16*16 pixels. The extracted feature maps were then patched into 512 patches, each 2*2 pixels. The patches were then reshaped into a single vector and then fed into the transformer encoder-decoder as is shown in the diagram below:

![Picture3](https://user-images.githubusercontent.com/63436458/184296166-136570df-60db-4bd2-9a9b-839bf263480b.png)

*Figure 5. Combined CNN and transformer architectures. The feature map from the fourth convolutional layer is patched and reformed into a sequence, which is then fed into the transformer encoder-decoder*

## Experiments and Results

As our primary evaluation metric, we will use the area under the receiver operating characteristic curve (AUROC). After visualizing the data and gathering basic statistics, we will properly perform data preprocessing. The initial plan is to start with a relatively simple CNNbased model, which will hopefully shed some light on potential problems and act as a setup for our final model. Then, we will be using pre-trained CNN-based models and try to customize or incrementally add complexity to those models to get some performance boost. As soon as our models become complex enough to overfit the datasets, we will introduce regularization and augment data if the need arises.

Finally, we will spend most of our time tuning the hyperparameters until we achieve low validation loss.
