# IDC IMAGE CLASSIFICATION USING DEEP LEARNING

This report uses deep learning techniques to correctly identify between IDC (-) and IDC (+) in breast cancer images using a public data set from Kaggle. The deep learning models covered in this report are a Convolution Neural network (CNN), a Convolutional recurrent network (CRNN), a Multilayer perceptron (MLP) and a pre-trained model (ResNet50). This report will cover the exploratory analysis, data preprocessing, model building, hyper-parameter tuning and In-depth evaluation of four Deep learning models.

### DATA IMPORT AND EXPLORATORY ANALYSIS
The dataset and its corresponding label were imported as a NumPy array by correctly specifying the file path and assigning them to variables X and Y. The labels associated with the images are 0 and 1, representing negative and positive IDC, respectively. The summary statistics obtained from the 'describeData' function show that the total number of images in the dataset is 5547, with 2759 as IDC (-) and 2788 as IDC (+), indicating an almost balanced target class. The shape of the image data was depicted to be 5547, 50, 50, 3, which means that all images are 50 X 50 by size with colours of red, blue and green (RGB) and thus corresponding to a 4D array.

Figures 1 and 2 show the pixel intensity plot and statistical measures of the image pixels, respectively. The IDC (+) class is not normally distributed and is skewed to the left. When combined, the standard deviation, maximum and minimum values of all the pixels conform to a normal distribution. Given its distribution, a total number of 4094 outliers were found, which might be attributed to the IDC (+) class.

![](https://github.com/odogwu25/Breast-cancer-images/blob/main/Images/Screenshot%202023-05-14%20at%2002.30.03.png)
