import tensorflow as tf
import os
import sys

import numpy as np
import pandas as pd

num_type = np.float32
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
print("Tensorflow version: {0}".format(tf.VERSION))


# Read CSV file
data_path = os.path.join("Data_Cortex_Nuclear.csv")
with open(data_path, "r") as r:
    df = pd.read_csv(data_path, dtype=str)


# Remove irrelevant or similar columns
to_exclude = {"MouseID", "Treatment", "Behavior", "Genotype"}
new_columns = set(df.columns) - to_exclude
df = df[list(new_columns)]
df = df.iloc[np.random.permutation(len(df))] #Shuffle samples


#Partition dataset
total_records = len(df)
train_records = total_records * 2 // 3
train_df = df.iloc[ :train_records ]
test_df = df.iloc[ train_records: ]


def convert_nan(iterable):
    return np.asarray( [num_type(0) if "nan" in str(element).lower() else num_type(element) for element in iterable] )

def tf_dataset(dataframe, label_column, label_dict):
    labels = np.zeros( (len(dataframe), len(label_dict)), dtype=num_type)
    for i in range(len(dataframe)):
        index = label_dict[ dataframe[label_column].iloc[i] ]
        labels[i][index] = 1


    dataframe = dataframe[ list(set(dataframe.columns) - {label_column}) ]

    features = np.ndarray( (len(dataframe), len(dataframe.columns)) , dtype=num_type )
    for i in range(len(dataframe.columns)):
        features[ : , i] = convert_nan( dataframe[dataframe.columns[i]] )

    dataset = tf.data.Dataset.from_tensor_slices( (features, labels) )
    return dataset



label_dict = {label : i for (i, label) in enumerate( set(train_df["class"]) ) }

print(len(train_df))
train_batch_size = 32
train_dataset = tf_dataset(train_df, "class", label_dict).batch(train_batch_size)

input_dim = int(train_dataset.output_shapes[0][1])
output_dim = len(label_dict)
def keras_model():
    return tf.keras.Sequential([
      tf.keras.layers.Dense(30, activation="relu", input_shape=(input_dim,)), 
      tf.keras.layers.Dense(30, activation="relu"),
      tf.keras.layers.Dense(output_dim, activation="softmax")
    ])


def cross_entropy_loss(y_hat, y):
    loss = -tf.reduce_sum( tf.log(y_hat) * y, axis=1 )
    return loss

def loss(model, x, y):
    return cross_entropy_loss(model(x), y)

def classification_accuracy(model, x, y):
     return tf.argmax( model(x), axis=1) == tf.argmax( y, axis = 1)


model = keras_model()

iterator = train_dataset.make_one_shot_iterator()
next_element = iterator.get_next()
loss_value = loss(model, next_element[0], next_element[1])

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss_value)
batch_loss = tf.reduce_mean(loss_value)

test_batch_size = 1
test_dataset = tf_dataset(test_df, "class", label_dict).batch(test_batch_size)

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    i = 0
    try:
       while True:
           [batch_loss_output, _] = sess.run( [batch_loss, train_op] )
           print("Batch {0}: Average loss = {1}".format(i, batch_loss_output), flush=True)
           i += 1
    except tf.errors.OutOfRangeError as e:
        pass
    
        


