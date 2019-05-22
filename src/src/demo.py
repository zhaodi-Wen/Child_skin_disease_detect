import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array,array_to_img
import numpy as np
import os
import random
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
# 初始化常量
batch_size = 32
class_num = 165
imgPath = './train/img'
labelPath = './train/label'
reg = '[^.]+'
test_img_path = './test/img'
test_label_path = './test/label'

# 定义函数将图片转为输入的矩阵
def get_img_matrix(path):
    img_array = img_to_array(load_img(path))
    #img_matrix = np.mat(img_array)
    return img_array

def get_train_data():
    #y_batch = np.zeros([batch_size, class_num])

    x_batch = np.zeros([batch_size, 256, 256, 3])
    skin_dieases_list = os.listdir(imgPath)
    y_batch = np.zeros([batch_size,class_num])
    for i in range(batch_size):
        # 随机选择一个皮肤病文件夹
        skin_dieases_seed = random.randint(0, len(skin_dieases_list)-1)
        img_list = os.listdir(imgPath+'/'+skin_dieases_list[skin_dieases_seed])
        # 随机选择皮肤病文件夹下的一张图片
        img_seed = random.randint(0, len(img_list)-1)
        x_batch[i] = get_img_matrix(imgPath+'/'+str(skin_dieases_list[skin_dieases_seed])+'/'+
                                    str(img_list[img_seed]))
        label_file = re.match(reg, str(img_list[img_seed])).group() + '.txt'
        #print(label_file)
        y_batch[i] = open(labelPath+'/'+str(skin_dieases_list[skin_dieases_seed])+'/'+
                          str(label_file)).readlines()
        #y_batch.append(skin_dieases_seed)

    return x_batch, y_batch

def get_test_data():

    skin_deseases_list = os.listdir(test_img_path)
    print(len(skin_deseases_list))
    test_x = np.zeros([len(skin_deseases_list),256,256,3])
    test_y = np.zeros([len(skin_deseases_list),class_num])
    for i in range(len(skin_deseases_list)):
        test_x[i] = (get_img_matrix(test_img_path+'/'+skin_deseases_list[i]))
        label_file = re.match(reg,str(skin_deseases_list[i])).group()+'.txt'
        test_y[i] = open(test_label_path+'/'+str(label_file)).readlines()

    return test_x, test_y

#test_x,test_y =get_test_data()
# 定义函数初始化权重矩阵
def weight_with_loss(shape,stddev):
    var = tf.Variable(tf.truncated_normal(shape,stddev))
    return var

# 定义损失函数
def get_loss(results,labels):
    labels = tf.cast(labels,tf.float32)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=results, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean




# 定义网络模型
# 输入数据&标签格式
x_input = tf.placeholder(tf.float32,[batch_size,256,256,3])
y_label = tf.placeholder(tf.int32,[batch_size,class_num])

# CNN Layer
# conv1
weight1 = weight_with_loss([5,5,3,64],stddev=0.05)
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))
conv1 = tf.nn.conv2d(x_input,weight1, [1,2,2,1],padding='SAME')
print("conv1 shape",conv1.shape)
relu1 = tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1 = tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
print("pool1 shape: ",pool1.shape)

# conv2
weight2 = weight_with_loss([5, 5, 64, 128], stddev=0.05)
bias2 = tf.Variable(tf.constant(0.0, shape=[128]))
conv2 = tf.nn.conv2d(pool1,weight2, [1, 2, 2, 1], padding='SAME')
print("conv2 shape", conv2.shape)
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2,ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
print("pool2 shape ", pool2.shape)

# FC Layer
# fc1 input: [batch_size,256,256,128]
reshape = tf.reshape(pool2, [batch_size, -1])
print("reshape shape", reshape.shape)
dim = reshape.get_shape()[1].value

weight_fc1 = weight_with_loss([dim, 384], stddev=0.04)
bias_fc1 = tf.Variable(tf.constant(0.0, shape=[384]))
local_fc1 = tf.nn.relu(tf.matmul(reshape, weight_fc1)+bias_fc1)
print("fc1 shape　", local_fc1.shape)

# output: [batch_size,384]
# fc2
weight_fc2 = weight_with_loss([384, 165], stddev=0.04)
bias_fc2 = tf.Variable(tf.constant(0.0, shape=[165]))
results = tf.add(tf.matmul(local_fc1, weight_fc2), bias_fc2)

loss = get_loss(results, y_label)
# 训练操作
train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
config = tf.ConfigProto(device_count={'gpu':0})
sess = tf.Session(config=config)
#tf.global_variables_initializer().run()
sess.run(tf.global_variables_initializer())
#print("results ", sess.run(results))
#tf.train.start_queue_runners()# 启动多线程加速

def train():
    max_iter_step = 1000
    for step in range(max_iter_step):
        np.set_printoptions(threshold=np.nan)

        x_batch, y_batch = get_train_data()
        #y_batch = tf.one_hot(y_batch,class_num,1,0)
        #y_batch = y_batch.eval(session=sess)
        #plt.imshow(array_to_img(x_batch[0]))
        #plt.show()

        _, loss_value = sess.run([train_op, loss], feed_dict={x_input: x_batch, y_label: y_batch})

        print('step %d,loss=%.8f' % (step, loss_value))
        if step % 50 == 0:

            correct_prediction = tf.equal(tf.argmax(results, 1), tf.argmax(y_label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("step {} accuracy {}".format(step,sess.run(accuracy, feed_dict={x_input: x_batch, y_label: y_batch})))



train()
















