#make_FTRecords

import os
import tensorflow as tf
from PIL import Image


cwd = os.getcwd()
print(cwd)

train_path = './train_new/img'
test_path = './test_new/img'
list = set(os.listdir(test_path))
classes=sorted(list,key=str.lower)
print(classes)

train_label = 'train.txt'
val_label = 'vat.txt'
train_writer = tf.python_io.TFRecordWriter("train.tfrecords")
test_writer = tf.python_io.TFRecordWriter("test.tfrecords")

def write_data(path,writer):
    for index,name in enumerate(classes):
        print(index,name)
        class_path = path+'/'+name+'/'
        print(class_path)
        for img_name in os.listdir(class_path):
            img_path = class_path+img_name
            print(img_path)
            img = Image.open(img_path)

            img = img.resize((224,224))
            img_raw = img.tobytes()
            with open(train_label,'a') as f:
                f.write(str(index)+'\n')



            # example = tf.train.Example(
            #     features=tf.train.Features(
            #         feature={
            #
            #             'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            #             "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            #
            #         }
            #     )
            # )
            #
            # writer.write(example.SerializeToString())


    writer.close()

write_data(train_path,train_writer)
# write_data(test_path,test_writer)

