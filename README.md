# Emotion_Detection

This small project uses the camera to detect faces and classify the emotion described by the facial characteristics.

OpenCV, which is a library of programming functions aimed at real-time computer vision, provided us the HAAR classifier for face detection.

We have implemented a 5 layer CNN for emotion detection. "ReLU" is used as the activation function. Average pooling size is 2x2 for all Convolutions.

First layer has 80 2D Convolutions with an 8x8 Kernel.
Second layer has 128 2D Convolutions with an 5x5 Kernel.
Third layer has 112 2D Convolutions with an 3x3 Kernel.
Final layer has 112 2D Convolutions with an 3x3 Kernel.

FER2013 dataset has been used for training and testing.
The final accuracy on the test set was 62.25%
