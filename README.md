# Rotten-Review-Class
An LSTM for predicting whether or not rotten tomatoes reviews are 'rotten' or 'fresh'
By Josh Khorsandi, Summer Junior Year Palisades Charter Highschool

## Dependencies:
python(3.7.8)

Tensorflow 2.0

Keras

Numpy

Matplotlib(for visualization)

## Using pretrained model 
There exists a pretrained model in saved_model.pbtxt(10 epochs, accuracy ~ 0.93, loss ~ 0.138)

To import the pretrained model, run: 

```python
import tensorflow as tf

tf.saved_model.load('/path/to/model/')
```

## Training your own model 

If you want to train your own model, edit 

```python 
history = model.fit(train_dataset, epochs=10, #edit to fit your own machine 
                    validation_data=test_dataset, 
                    validation_steps=30)
```
then run 

```bash
python main.py
```
Main will save the model to a folder named tomato1 in the same directory

## Dataset
The dataset I used is from 2 csv files, 1 encodes a 'fresh' review while 0 encodes a 'rotten' review. There are 2 csv files with 480,000 reviews total. 'rt_reviews.csv' is for training and 'eval.csv' is for testing

