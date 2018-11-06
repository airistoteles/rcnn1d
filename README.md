# rcnn1d
Recurrent Convolutional Neural Network Class - (almost) scikit-learn API

This module contains one-dimensional (Recurrent) Convolutional Neural Networks - Classifiers.

One-dimensional as in Keras' definition of Conv1D. Kernelsize is always (num_channels, kernel_width).

    -> Convolution only in one direction

    -> channel order doesn't matter

    -> but because of this channel order information is lost

It can be used to classify one-class-problems like epileptic seizure prediction using (raw) EEG data.
Conventional CNN as well as R-CNN are implemented.
Therefore the tied_layers class by Nico Hoffmann (https://github.com/nih23/UKDDeepLearning/tree/master/FunctionalImaging) 
was used to implement a Recurrent ConvLayer of the size 4 as in https://doi.org/10.1109/CVPR.2015.7298958

See code (rcnn.py) for further documentation

# Install needed libraries
... the easy way. If you build Tensorflow from source it trains much faster.

```
pip install numpy
pip install tensorflow 
pip install keras
pip install scikit-learn
```

# Example
```python
import rcnn
import numpy as np

# dummy EEG dataset: 10 samples, 16 channels, 3000 EEG data samples per sample
x = np.random.random([10, 16, 3000])
y = np.zeros([10,])
y[0] = 1
x_val = np.random.random([10, 16, 3000])
y_val = np.zeros([10,])
y_val[0] = 1

# init model
model = rcnn.RecCnnRCNN_generic(recurrent=True, num_features=8, conv_depth=8, save_model=False)
# fit model for 2 epochs
model.fit(x, y, x_val, y_val, epochs=2) 

# evaluate model using Area Under Curve (AUC)
print("AUC=" + str(model.score(x_val, y_val)))
```
Results in:
```
Using TensorFlow backend.

Train on 10 samples, validate on 10 samples

Epoch 1/2
10/10 [==============================] - 15s 2s/step - loss: 1.2516 - binary_accuracy: 0.8000 - val_loss: 0.7368 - val_binary_accuracy: 0.2000

Epoch 2/2
10/10 [==============================] - 0s 5ms/step - loss: 1.1428 - binary_accuracy: 0.8000 - val_loss: 0.6593 - val_binary_accuracy: 0.6000

AUC=0.7777777777777778
```
