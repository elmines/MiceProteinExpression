import tensorflow as tf
import os
import sys

import numpy as np
import pandas as pd

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
print("Tensorflow version: {0}".format(tf.VERSION))

to_exclude = {"MouseID", "Treatment", "Behavior", "Genotype"}

data_path = os.path.join("Data_Cortex_Nuclear.csv")

num_type = np.float32


with open(data_path, "r") as r:
    df = pd.read_csv(data_path, dtype=str)

new_columns = set(df.columns) - to_exclude

df = df[list(new_columns)]
df = df.iloc[np.random.permutation(len(df))] #Shuffle samples


total_records = len(df)
train_records = total_records * 2 // 3
test_records = total_records - train_records

train_df = df.iloc[ :train_records ]
test_df = df.iloc[ train_records: ]

label_dict = {label : i for (i, label) in enumerate( set(train_df["class"]) ) }


def convert_nan(iterable):
    return np.asarray( [num_type(0) if "nan" in str(element).lower() else num_type(element) for element in iterable] )

def tf_dataset(dataframe, label_column, label_dict):
    print("label_dict =", label_dict)

    for class_name in dataframe[label_column]:
        print("%s --> %s" % (class_name, label_dict[class_name]) )
    #sys.exit(0)

    labels = np.asarray([ label_dict[class_name] for class_name in dataframe[label_column] ], dtype=num_type)
    print("labels =", labels)
    #sys.exit(0)
    dataframe = dataframe[ list(set(dataframe.columns) - {label_column}) ]

    features = np.ndarray( (len(dataframe), len(dataframe.columns)) , dtype=num_type )
    for i in range(len(dataframe.columns)):
        features[ : , i] = convert_nan( dataframe[dataframe.columns[i]] )

    dataset = tf.data.Dataset.from_tensor_slices( (features, labels) )
    return dataset



short_train = train_df.iloc[:10]

print("label column=")
print(short_train["class"])

short_dataset = tf_dataset(short_train, "class", label_dict)


iterator = short_dataset.make_one_shot_iterator()
pair = iterator.get_next()
print(pair)


with tf.Session() as sess:
    features, label = pair
    dataset_features = features.eval()
    class_label = label.eval()
    

print(short_train.iloc[0])
print("DATASET OUTPUT:")
print(dataset_features)
print(class_label)
print(label_dict)

