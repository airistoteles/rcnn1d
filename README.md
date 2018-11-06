# rcnn1d
Recurrent Convolutional Neural Network Class - (almost) scikit-learn API

This module contains (Recurrent) Convolutional Neural Networks - Classifiers

It can be used to classify one-class-problems like epileptic seizure prediction using (raw) EEG data.
Conventional CNN as well as R-CNN are implemented.
Therefor the tied_layers class by Nico Hoffmann (https://github.com/nih23) were used to implement a Recurrent ConvLayer
of the size 4 as in https://doi.org/10.1109/CVPR.2015.7298958

See code (rcnn.py) for further documentation

# Usage
```python

# your data goes here
x, y, x_val, y_val = … 

# init model
model = rcnn.RecCnnRCNN_generic(recurrent=True, num_features=8, conv_depth=8, save_model=False) # init model
# fit model for 2 epochs
model.fit(x, y, x_val, y_val, epochs=2) 

# evaluate model using Area Under Curve (AUC)
print("AUC=" + str(model.score(x_val, y_val)))
```
