import tensorflow as tf
print("Tensorflow version: {0}".format(tf.VERSION))
import os
import sys
from pydoc import locate #For determining our TensorFlow numeric types from our Numpy ones
import numpy as np
import pandas as pd

#Seed random number generators
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

#Set numeric types for operations
num_type = np.float64
tf_num_type = tf.as_dtype(num_type)

#Minibatch Descent Hyperparameters
minibatch_descent = False
train_batch_size = 64 #Only used if minibatch_descent is True

batch_ratio = 0.5 #Let each batch compose half the dataset

valid_batch_size = 32
num_epochs = 100


# Read CSV file
data_path = os.path.join("Data_Cortex_Nuclear.csv")
with open(data_path, "r") as r:
    df = pd.read_csv(data_path, dtype=str)
# Remove irrelevant or similar columns
to_exclude = {"MouseID", "Treatment", "Behavior", "Genotype"}
new_columns = set(df.columns) - to_exclude
df = df[list(new_columns)]

df = df.iloc[np.random.permutation(len(df))] #Shuffle samples

#Partition dataset - 80% training, 10% validation, %10 testing
num_samples = len(df)
num_training = num_samples * 8 // 10
if not minibatch_descent: # Scale batch sizes according to dataset size
    train_batch_size = int(num_training * batch_ratio) if batch_ratio else num_training

num_validation = num_samples // 10
num_testing = num_samples - num_training - num_validation
train_df = df.iloc[ :num_training ]
validation_df = df.iloc[ num_training : num_training + num_validation ]
test_df = df.iloc[ num_training + num_validation : ]
print("{0} samples: {1} training, {2} validation, {3} testing".format(num_samples, num_training, num_validation, num_testing), flush=True)



def convert_nan(iterable):
    return np.asarray( [num_type(0) if "nan" in str(element).lower() else num_type(element) for element in iterable], dtype=num_type )

def tf_dataset(dataframe, label_column, label_dict):
    labels = np.zeros( (len(dataframe), len(label_dict)), dtype=num_type)
    for i in range(len(dataframe)):
        index = label_dict[ dataframe[label_column].iloc[i] ]
        labels[i][index] = 1

    dataframe = dataframe[ list(set(dataframe.columns) - {label_column}) ]
    features = np.ndarray( (len(dataframe), len(dataframe.columns)) , dtype=num_type )
    for i in range(len(dataframe.columns)):
        features[ : , i] = convert_nan( dataframe[dataframe.columns[i]] )

    return tf.data.Dataset.from_tensor_slices( (features, labels) )



label_dict = {label : i for (i, label) in enumerate( set(train_df["class"]) ) }
train_dataset = tf_dataset(train_df, "class", label_dict).batch(train_batch_size)
test_dataset = tf_dataset(test_df, "class", label_dict).batch(1)

input_dim = int(train_dataset.output_shapes[0][1])
output_dim = len(label_dict)
def keras_model():
    return tf.keras.Sequential([
      tf.keras.layers.Dense(30, activation="relu", input_shape=(input_dim,), dtype=tf_num_type), 
      tf.keras.layers.Dense(30, activation="relu", dtype=tf_num_type),
      tf.keras.layers.Dense(output_dim, activation="softmax", dtype=tf_num_type)
    ])


def cross_entropy_loss(y_hat, y):
    return -tf.reduce_sum( tf.log(y_hat) * y, axis=1 )
def loss(model, x, y):
    return cross_entropy_loss(model(x), y)
def classification_accuracy(model, x, y):
     return tf.argmax( model(x), axis=1) == tf.argmax( y, axis = 1)


iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
training_init_op = iterator.make_initializer(train_dataset)
next_element = iterator.get_next()

model = keras_model()
loss_value = loss(model, next_element[0], next_element[1])
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss_value)
batch_loss = tf.reduce_mean(loss_value)


def num_batches(num_samples, batch_size):
    return num_samples // batch_size + (0 if num_samples % batch_size == 0 else 1)

train_batches = num_batches(num_training, train_batch_size)
print("Training on %d samples divided into batches of %d" % (num_training, train_batches))
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    for epoch in range(num_epochs):
        sess.run( training_init_op )
        print("EPOCH {0}".format(epoch))
        for i in range(train_batches):
            [batch_loss_output, _] = sess.run( [batch_loss, train_op] )
            print("\tBatch {0}: Average loss = {1}".format(i, batch_loss_output), flush=True)
