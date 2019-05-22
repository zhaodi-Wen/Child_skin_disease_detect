import tensorflow as tf
path="new_train.pb"        #pb文件位置和文件名
inputs=["inputs"]               #模型文件的输入节点名称
classes=["classes"]            #模型文件的输出节点名称
#converter = tf.contrib.lite.TocoConverter.from_frozen_graph(path, inputs, classes)
tflite_model=converter
open("./model_pb.tflite", "wb").write(tflite_model)
