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

    #labels = np.asarray([ label_dict[class_name] for class_name in dataframe[label_column] ], dtype=num_type)

    labels = np.zeros( (len(dataframe), len(label_dict)), dtype=num_type)
    for i in range(len(dataframe)):
        index = label_dict[ dataframe[label_column].iloc[i] ]
        labels[i][index] = 1


    dataframe = dataframe[ list(set(dataframe.columns) - {label_column}) ]

    features = np.ndarray( (len(dataframe), len(dataframe.columns)) , dtype=num_type )
    for i in range(len(dataframe.columns)):
        features[ : , i] = convert_nan( dataframe[dataframe.columns[i]] )

    #print(features.shape)

    dataset = tf.data.Dataset.from_tensor_slices( (features, labels) )
    return dataset



label_dict = {label : i for (i, label) in enumerate( set(train_df["class"]) ) }
short_df = train_df.iloc[:96]

train_batch_size = 32
train_dataset = tf_dataset(short_df, "class", label_dict).batch(train_batch_size)

input_dim = int(train_dataset.output_shapes[0][1])
output_dim = len(label_dict)
def keras_model():
    return tf.keras.Sequential([
      tf.keras.layers.Dense(30, activation="relu", input_shape=(input_dim,)), 
      tf.keras.layers.Dense(30, activation="relu"),
      #tf.keras.layers.Dense(output_dim)
      tf.keras.layers.Dense(output_dim, activation="softmax")
    ])

def one_hot_vector(num_clases, indices):
    vector = np.zeros(size, dtype=num_type)
    vector[index] = 1

def cross_entropy_loss(y_hat, y, axis=1):
    loss = -tf.reduce_sum( tf.log(y_hat) * y, axis=axis )
    return loss

def loss(model, x, y):
    return cross_entropy_loss(model(x), y)


"""
y_hat = np.asarray( [ [0.3, 0.3, 0.4], [0.7, 0.15, 0.15] ], dtype=num_type)
y     = np.asarray( [ [0,   0,   1  ], [1,   0,    0   ] ], dtype=num_type)
loss = cross_entropy_loss(model, y_hat, y)
"""

def running_average(curr_avg, num_samples, new_sample):
    return ( curr_avg * num_samples + new_sample) / (num_samples + 1)


model = keras_model()
iterator = train_dataset.make_one_shot_iterator()
next_element = iterator.get_next()
loss_value = loss(model, next_element[0], next_element[1])
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss_value)

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    #uninitialized = sess.run( tf.report_uninitialized_variables() )
    #print(uninitialized)

    try:
       while True:
           sess.run(train_op)
           """
           features, label = pair
           print(features)
           print(label)
           model_output = model()
           print(model_output)
           #print(model_output.eval())
           break
           gradients = loss_and_grad(model, features, label)
           optimizer.apply_gradients( zip(gradients, model.variables), global_step = tf.train.get_or_create_global_step() )
           """
    except tf.errors.OutOfRangeError:
        pass
    
        
