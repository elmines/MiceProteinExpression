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
df = df.iloc[np.random.permutation(len(df))]

def parse_csv(line):
    return line

total_records = tf.constant(len(df))
train_records = total_records * 2 // 3
test_records = total_records - train_records

with tf.Session() as sess:
    print(train_records.eval())
    print(test_records.eval())

"""
full_dataset = tf.data.TextLineDataset(data_path).skip(1).map(parse_csv).shuffle(2000)
print(full_dataset)


train_dataset, test_dataset = tf.split(full_dataset, [train_records, test_records], axis = 0)

#print("Dataset: {0}".format(data_set_fp))
"""
