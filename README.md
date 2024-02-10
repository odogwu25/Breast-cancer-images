# IDC IMAGE CLASSIFICATION USING DEEP LEARNING

This report uses deep learning techniques to correctly identify between IDC (-) and IDC (+) in breast cancer images using a public data set from Kaggle. The deep learning models covered in this report are a Convolution Neural network (CNN), a Convolutional recurrent network (CRNN), a Multilayer perceptron (MLP) and a pre-trained model (ResNet50). This report will cover the exploratory analysis, data preprocessing, model building, hyper-parameter tuning and In-depth evaluation of four Deep learning models.

### DATA IMPORT AND EXPLORATORY ANALYSIS
The dataset and its corresponding label were imported as a NumPy array by correctly specifying the file path and assigning them to variables X and Y. The labels associated with the images are 0 and 1, representing negative and positive IDC, respectively. The summary statistics obtained from the 'describeData' function show that the total number of images in the dataset is 5547, with 2759 as IDC (-) and 2788 as IDC (+), indicating an almost balanced target class. The shape of the image data was depicted to be 5547, 50, 50, 3, which means that all images are 50 X 50 by size with colours of red, blue and green (RGB) and thus corresponding to a 4D array.

Figures 1 and 2 show the pixel intensity plot and statistical measures of the image pixels, respectively. The IDC (+) class is not normally distributed and is skewed to the left. When combined, all the pixels' standard deviation, maximum and minimum values conform to a normal distribution. Given its distribution, 4094 outliers were found, which might be attributed to the IDC (+) class.

![](https://github.com/odogwu25/Breast-cancer-images/blob/main/Images/Screenshot%202023-05-14%20at%2002.30.03.png)

### DATA PREPROCESSING

Deep learning models do not need much preprocessing compared to classical machine learning models, as they can accept raw data to be fed into the model while maintaining a high accuracy level. However, some preprocessing, such as image resizing, ensures that all images are set to a fixed size, normalizes, and deals with outliers. The images were resized to a fixed size to maintain consistency while retaining their original shape. Two standardization techniques were carried out on each model to compare which would work best because relying on one is not a rule of thumb. The min-max scaler, which is achieved by dividing the pixel values by 255 (the total number of possible intensity values a coloured image can have), will ensure that the pixel values of the images are set to a fixed range between 0 and 1 OR -1 and 1, while the standard scaler will transform data to have zero mean and unit variance. These standardisation forms help ensure that the model can learn the underlying patterns in the data more effectively. Outliers found were handled by setting their values to the mean of non-outlier pixels.

Image augmentation was performed on the training set to enhance the number of images used to train the model, as this will aid the model's generalization of unseen data. The data was divided into train and test partitions to validate the model after training with an 80:20 ratio (4437 images for training and 1110 for testing). Finally, the num_classes variable is set to 1 due to the binary classification task. The model output will be a single scalar value between 0 and 1, representing the predicted probability of the positive class.

### MODEL SELECTION & ARCHITECTURE

Due to its fast and accurate feature extraction function and comprehensive trainable network architecture, the convolutional neural network (CNN) has successfully classified images (Lei et al., 2019). Therefore, this report will use CNN as the first choice but will be compared with the multilayer perceptron (MLP), a feed-forward neural network and CRNN, a hybrid model (CNN + RNN). The CNN, CRNN, MLP and ResNet (pre-trained CNN) models were built and fine-tuned based on the following:

**- Convolutional layers:** This layer extracts low-level features using one filter across the kernel and then applying an activation function “relu”. The kernels go around the images, executing convolutions on local image data to create feature maps (Xiao et al., 2020). Stride was set as 1 by default to cause overlapping if the kernel missed any feature in its first movement.
**- Pooling Layers:** Captures mid-level features by downsizing the feature maps to prevent overfitting.
**- Dense Layer:** The fully connected layer specifies the number of classes for the task and an activation function that suits the task. Binarycrossentropy is used as the final output due to the binary classification problem.
** - Learning Rate:** A hyperparameter determines how quickly or slowly a model learns.
**- Batch size:** Refers to the number of training samples that the neural network uses during
each forward and backward pass during training.
**- Epoch:** A full pass or iteration through the entire training dataset. The more epochs there
are, the more time the model has to learn.
**-Dropout:** This works by setting some percentage of the neuron connection to zero. This is
done only in the training so that the neural network doesn’t memorize the training data to generalize properly.



