# Convolution Neural Network - Emotion Recognition
This is a 2D-CNN approache for real time emotion recognition using web camera. This CNN include 4 convolution layers and 2 fully connected layers. Final layer has 7 nodes which produce seven emotion (Anger, Disgust, Fear, Happy, Sadness, Surprise, Neutral) values indepedantly between 0-1.
Fer2013 emotion dataset use for training and it devide into 3 sets such as training (80%), validation (10%) and testing (10%) sets.
You can download easily Fer2013 data set from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Download the data set dand extract it into dataset folder.
This take about 8 hours to train using GeForce 940MX 2GB and 8GB RAM, Core i5 processor.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 48, 48, 64)        640       
_________________________________________________________________
batch_normalization_1 (Batch (None, 48, 48, 64)        256       
_________________________________________________________________
activation_1 (Activation)    (None, 48, 48, 64)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 24, 24, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 24, 24, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 128)       204928    
_________________________________________________________________
batch_normalization_2 (Batch (None, 24, 24, 128)       512       
_________________________________________________________________
activation_2 (Activation)    (None, 24, 24, 128)       0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 12, 12, 128)       0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 12, 12, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 512)       590336    
_________________________________________________________________
batch_normalization_3 (Batch (None, 12, 12, 512)       2048      
_________________________________________________________________
activation_3 (Activation)    (None, 12, 12, 512)       0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 512)         0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 6, 6, 512)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 6, 512)         2359808   
_________________________________________________________________
batch_normalization_4 (Batch (None, 6, 6, 512)         2048      
_________________________________________________________________
activation_4 (Activation)    (None, 6, 6, 512)         0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 3, 3, 512)         0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 3, 3, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               1179904   
_________________________________________________________________
batch_normalization_5 (Batch (None, 256)               1024      
_________________________________________________________________
activation_5 (Activation)    (None, 256)               0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               131584    
_________________________________________________________________
batch_normalization_6 (Batch (None, 512)               2048      
_________________________________________________________________
activation_6 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 7)                 3591      
=================================================================
Total params: 4,478,727
Trainable params: 4,474,759
Non-trainable params: 3,968
_________________________________________________________________

Accuracy on Test set :  0.6544998606854276


