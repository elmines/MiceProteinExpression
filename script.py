import tensorflow as tf
import os

print("Tensorflow version: {0}".format(tf.VERSION))

data_path = os.path.join("Data_Cortex_Nuclear.csv")
#data_set_fp = tf.keras.utils.get_file(fname=data_path)

def parse_csv(line):
    return line

full_dataset = tf.data.TextLineDataset(data_path).skip(1).map(parse_csv).shuffle(2000)
print(full_dataset)

total_records = tf.shape(full_dataset)[0]
train_records = total_records * 2 // 3
test_records = total_records - train_records

train_dataset, test_dataset = tf.split(full_dataset, [train_records, test_records], axis = 0)

#print("Dataset: {0}".format(data_set_fp))

