import tensorflow as tf
import os
import sys

import numpy as np
import pandas as pd

print("Tensorflow version: {0}".format(tf.VERSION))

to_exclude = {"MouseID", "Treatment", "Behavior", "Genotype"}

data_path = os.path.join("Data_Cortex_Nuclear.csv")


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

def tf_dataset(dataframe, label_column, label_dict):
    labels = np.asarray([ label_dict[class_name] for class_name in dataframe[label_column] ], dtype=np.float32 )

    print("NEW:")
    dataframe = dataframe[ list(set(dataframe.columns) - {label_column}) ]
    print(dataframe)
    print(labels)

    for column in dataframe.columns: 


tf_dataset(train_df.iloc[:10], "class", label_dict)


#train_dataset = tf_dataset(train_df)
"""
with tf.Session() as sess:
    print(train_records.eval())
    print(test_records.eval())

full_dataset = tf.data.TextLineDataset(data_path).skip(1).map(parse_csv).shuffle(2000)
print(full_dataset)


train_dataset, test_dataset = tf.split(full_dataset, [train_records, test_records], axis = 0)

#print("Dataset: {0}".format(data_set_fp))
"""
