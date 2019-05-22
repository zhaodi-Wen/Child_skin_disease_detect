'''普通CNN'''

import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import os
import random
import re

# 初始化常量
batch_size =8
class_num = 10
imgPath = './img'
labelPath = './label'
trainPath = './train'
reg = '[^.]+'


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string("img_path",'./test_new/img/acanthosis-nigricans/acanthosis-nigricans_0_69.jpg',"direction to tested image")
#np.set_printoptions(suppress=False)

# 定义函数将图片转为输入的矩阵
'''def get_img_matrix(path):
    img_array = img_to_array(load_img(path))
    # 归一化
    img_array = img_array/255.
    return img_array'''

# 获取训练数据
def get_data():
    y_batch = np.zeros([batch_size])
    x_batch = np.zeros([batch_size,256,256,3],dtype=float)
    skin_dieases_list = os.listdir(imgPath)
    for i in range(batch_size):
        # 随机选择一个皮肤病文件夹
        skin_dieases_seed = random.randint(0,class_num-1)
        # debug
        #skin_dieases_seed = 1

        train_list = os.listdir(trainPath+'/txt/'+skin_dieases_list[skin_dieases_seed])
        # 随机选择皮肤病文件夹下的一张图片
        train_seed = random.randint(0,len(train_list)-1)
        # debug
        #train_seed = 1

        x_batch[i] = np.resize(np.loadtxt(trainPath+'/txt/'+str(skin_dieases_list[skin_dieases_seed])+'/'+
                                    str(train_list[train_seed])),new_shape=(256,256,3))
        # 检查是否有nan
        np.any(np.isnan(x_batch[i]))

        #y_batch[i] = np.loadtxt(labelPath+'/'+str(skin_dieases_list[skin_dieases_seed])+'/'+
        #                  str(train_list[train_seed]))[0:class_num]
        y_batch[i] = skin_dieases_seed
    return x_batch,y_batch


# 定义函数初始化权重矩阵
def weight_with_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name = "weight_loss")  #L2范数正则化
        tf.add_to_collection("losses",weight_loss)  #将变量weight_loss放入losses变成一个列表
    return var

# 定义损失函数
def get_loss(results,labels):
    labels = tf.cast(labels,tf.int32)
    results = tf.cast(results,tf.float32)
    #cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(results,1e-8,1.0)))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=results,labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #tf.add_to_collection("losses", cross_entropy)
    #return tf.add_n(tf.get_collection("losses"), name="total_loss")
    return cross_entropy_mean

sess = tf.InteractiveSession()
# 定义网络模型
# 输入数据&标签格式
x_input = tf.placeholder(tf.float32,[batch_size,256,256,3])
y_label = tf.placeholder(tf.int32,[batch_size])

# CNN Layer
# conv1
weight1 = weight_with_loss([5,5,3,32],stddev=0.05,w1=0.0)
bias1 = tf.Variable(tf.constant(0.0,shape=[32]))
conv1 = tf.nn.conv2d(x_input,weight1,[1,1,1,1],padding='SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1,bias1))
norm1 = tf.nn.lrn(relu1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
pool1 = tf.nn.max_pool(norm1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

# conv2
weight2 = weight_with_loss([5,5,32,32],stddev=0.05,w1=0.0)
bias2 = tf.Variable(tf.constant(0.0,shape=[32]))
conv2 = tf.nn.conv2d(pool1,weight2,[1,2,2,1],padding='SAME')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2,bias2))
norm2 = tf.nn.lrn(relu2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

# conv3
weight3 = weight_with_loss([5,5,32,64],stddev=0.05,w1=0.0)
bias3 = tf.Variable(tf.constant(0.0,shape=[64]))
conv3 = tf.nn.conv2d(pool2,weight3,[1,1,1,1],padding='SAME')
relu3 = tf.nn.relu(tf.nn.bias_add(conv3,bias3))
norm3 = tf.nn.lrn(relu3,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
pool3 = tf.nn.max_pool(norm3,ksize=[1,3,3,1],strides=[1,4,4,1],padding='SAME')

# FC Layer
# fc1 input:
reshape = tf.reshape(pool3,[batch_size,-1])
dim = reshape.get_shape()[1].value

weight_fc1 = weight_with_loss([dim,64],stddev=0.04,w1=0.004)
bias_fc1 = tf.Variable(tf.constant(0.0,shape=[64]))
local_fc1 = tf.nn.relu(tf.matmul(reshape,weight_fc1)+bias_fc1)

# output: [batch_size,384]
# fc2
weight_fc2 = weight_with_loss([64,class_num],stddev=1/10,w1=0.0)
bias_fc2 = tf.Variable(tf.constant(0.0,shape=[class_num]))
# debug
before = tf.add(tf.matmul(local_fc1,weight_fc2),bias_fc2)
results = tf.add(tf.matmul(local_fc1,weight_fc2),bias_fc2)

# debug----------
# 用卷积代替全连接
'''k_num = 64
size = pool3.get_shape()[1].value
channel = pool3.get_shape()[3].value
weight4 = weight_with_loss([size,size,channel,k_num],stddev=0.05,w1=0.0)
bias4 = tf.Variable(tf.constant(0.0,shape=[64]))
conv4 = tf.nn.conv2d(pool3,weight4,[1,size,size,1],padding='SAME')
local = tf.nn.bias_add(conv4,bias4)

reshape = tf.reshape(local,[batch_size,-1])
dim = reshape.get_shape()[1].value
weight_fc = weight_with_loss([dim,class_num],stddev=1/10,w1=0.0)
bias_fc = tf.Variable(tf.constant(0.0,shape=[class_num]))
results = tf.nn.softmax(tf.add(tf.matmul(reshape,weight_fc),bias_fc))'''



# ---------------

loss = get_loss(results,y_label)
# 训练操作
train_op = tf.train.AdamOptimizer(0.002).minimize(loss)


tf.global_variables_initializer().run()
tf.train.start_queue_runners()# 启动多线程加速

# 预测评估
top_k_results = tf.nn.top_k(results,1)
#top_k_labels = tf.nn.top_k(y_label,1)

saver = tf.train.Saver()

def train():
    max_iter_step = 1000
    target_loss = 1
    for step in range(max_iter_step):
        x_batch,y_batch = get_data()
        true_count = 0
        _,loss_value,output = sess.run([train_op,loss,top_k_results],feed_dict={x_input:x_batch,y_label:y_batch})
        #print(output)
        output = output[1]
        #print(output)
        #print(y_batch)
        #print(debug2)
        '''print(output)
        print(labels)
        print(debug)
        print(np.array(debug).shape)'''
        #sess.run(tf.Print(results,[results]),feed_dict={x_input:x_batch,y_label:y_batch})
        for i in range(batch_size):
            if output[i]==y_batch[i]:
                true_count+=1
        precision = float(true_count / batch_size)
        if step%5 == 0:
            print('step %d,loss=%.4f,predition=%.4f' %(step,loss_value,precision))
            print(np.array(output).reshape([1,batch_size]))
            print(y_batch)
        #if loss_value<target_loss:
        #    break




# 预测，
def predict(*args):
    skin_dieases_list = open('./skin_dieases_list.txt').readlines()
    for i in range(len(skin_dieases_list)):
        skin_dieases_list[i] = skin_dieases_list[i].strip('\n')
    img_path = FLAGS.img_path
    y_batch = np.zeros([batch_size])
    x_batch = np.zeros([batch_size, 256, 256, 3], dtype=float)
    img = img_to_array(load_img(img_path))
    img = img/255.
    img = np.resize(img,new_shape=(256,256,3))
    #print(img)
    x_batch[0] = img
    saver = tf.train.import_meta_graph(model_path+'.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model1'))
    output = sess.run([top_k_results],feed_dict={x_input:x_batch,y_label:y_batch})
    print(output)
    output = output[0][1]
    print(output)
    print('predict skin diease is: ' + skin_dieases_list[int(output[0])])

# 训练模型
#train()
# 保存模型
model_path = './model1/model.ckpt'
saver.save(sess, model_path)
saver_pb = tf.train.write_graph(sess.graph_def, './model1', 'train.pb', as_text=False)

if __name__ =='__main__':


    #predict_img = 'F:\大创\Demo\test\label\acne-conglobata_0_571.jpg'
    predict()
#predict()
















