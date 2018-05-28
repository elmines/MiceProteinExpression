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

# Model saving options
models_dir = "./models/"
best_model_prefix = models_dir + "best_model"
last_model_prefix = models_dir + "model"
save_latest = True

#HYPERPARAMETERS
#  Training
num_epochs = 50
#    Minibatch Descent
minibatch_descent = True
train_batch_size = 64 #Only used if minibatch_descent is True
batch_ratio = 0.25 #Let each batch compose batch_ratio of the dataset (i.e. 1/4, 1/2, etc.)
#  Validation
valid_batch_size = 64 #Initialize to total number of validation samples later
max_stalled_steps = 5
loss_epsilon = 0.0001 #Epsilon used when comparing validation loss metrics for equality
#  Testing
test_batch_size = 64


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
num_train = num_samples * 8 // 10
if not minibatch_descent: # Scale batch sizes according to dataset size
    train_batch_size = int(num_train * batch_ratio) if batch_ratio else num_train

num_valid = num_samples // 10
if valid_batch_size > num_valid: valid_batch_size = num_valid

num_test = num_samples - num_train - num_valid

train_df = df.iloc[ :num_train ]
valid_df = df.iloc[ num_train : num_train + num_valid ]
test_df = df.iloc[ num_train + num_valid : ]
print("{0} samples: {1} training, {2} validation, {3} testing".format(num_samples, num_train, num_valid, num_test), flush=True)



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
valid_dataset = tf_dataset(valid_df, "class", label_dict).batch(valid_batch_size)
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
def classification_error(model, x, y):
     return tf.argmax( model(x), axis=1) != tf.argmax( y, axis = 1)


# Common tensors
model = keras_model()
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()
loss_value = loss(model, features, labels)

# Training tensors
train_init_op = iterator.make_initializer(train_dataset)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss_value)

# Validation tensors
valid_init_op = iterator.make_initializer(valid_dataset)

# Testing tensors
test_init_op = iterator.make_initializer(test_dataset)

def num_batches(num_samples, batch_size):
    return num_samples // batch_size + (0 if num_samples % batch_size == 0 else 1)

def update_loss(epoch_loss, epoch_samples, batch_loss, batch_samples):
    new_loss = (epoch_loss*epoch_samples + batch_loss*batch_size) / (epoch_samples + batch_size)
    new_samples = batch_loss + batch_samples
    return (new_loss, new_samples)


saver = tf.train.Saver() #Saver for all global variables
train_batches = num_batches(num_train, train_batch_size)
valid_batches = num_batches(num_valid, valid_batch_size)
print("Training on %d samples divided into batches of %d" % (num_train, train_batch_size))
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )

    best_valid_loss = float("inf")
    stalled_steps = 0
    for epoch in range(num_epochs):
        if stalled_steps >= max_stalled_steps:
            print("Validation has stalled for {0} steps, ending training . . .".format(stalled_steps))
            break
        sess.run( train_init_op )
        epoch_loss = 0.0
        epoch_samples = 0

        print("EPOCH {0}".format(epoch), flush=True)
        for i in range(train_batches):
            [sample_losses, _] = sess.run( [loss_value, train_op] )
            batch_size = len(sample_losses)
            batch_loss = np.mean(sample_losses)
            print("\tBatch {0}: {1} samples, average loss = {2}".format(i, batch_size, batch_loss), flush=True)
            #Keep running loss average
            (epoch_loss, epoch_samples) = update_loss(epoch_loss, epoch_samples, batch_loss, batch_size)
        print("\tAverage loss for epoch = {0}".format(epoch_loss), flush=True)

        sess.run( valid_init_op )
        valid_loss = 0.0
        valid_samples = 0
        print("\tVALIDATION SET")
        for i in range(valid_batches):
            sample_losses = sess.run(loss_value)
            batch_size = len(sample_losses)
            batch_loss = np.mean(sample_losses)
            (valid_loss, valid_samples) = update_loss(valid_loss, valid_samples, batch_loss, batch_size)
            print("\t\tBatch {0}: {1} samples, average loss = {2}".format(i, batch_size, batch_loss), flush=True)
        print("\t\tAverage validation loss = {0}".format(valid_loss), flush=True)

        if  (valid_loss > best_valid_loss) or (abs(valid_loss - best_valid_loss) < loss_epsilon):
            stalled_steps += 1
        else:
            best_valid_loss = valid_loss
            stalled_steps = 0
            saver.save(sess, best_model_prefix)
            print("\t\tNew best valididation loss: saved latest model to {}".format(best_model_prefix), flush=True)

        if save_latest:
            saver.save(sess, last_model_prefix)
            print("\tSaved latest model to {}".format(last_model_prefix), flush=True)
           

def test_model(model_prefix):
    test_batches = num_batches(num_test, test_batch_size)
    print("Testing model {0} on {1} samples divided into batches of {2}".format(model_prefix, num_test, test_batch_size))
    with tf.Session() as sess:
        saver.restore(sess, model_prefix)
        sess.run(test_init_op)
        test_losses = []
        for i in range(test_batches):
            sample_losses = sess.run(loss_value)
            for loss_val in sample_losses:
                test_losses.append(loss_val)
        average_test_loss = sum(sample_losses) / len(sample_losses)
        print("Average test loss = {0}".format(average_test_loss))

test_model(best_model_prefix)
test_model(last_model_prefix)
