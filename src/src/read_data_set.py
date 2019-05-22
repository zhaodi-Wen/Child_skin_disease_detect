import os
import tensorflow as tf
from PIL import Image
import time
tf.device('/gpu:3')
import tempfile
import subprocess

tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess
from tensorflow.python.framework import graph_util

'''
cwd = os.getcwd()
filename_queue = tf.train.string_input_producer(["train_own.tfrecords"])

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)
    }
)

image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [256, 256, 3])

label = tf.cast(features['label'], tf.int32)
img_batch = tf.train.shuffle_batch([image],batch_size=4,num_threads=8,
                                   capacity=112,
                                   min_after_dequeue=100)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(100):

        example,l = sess.run([image,label])
        print(example.shape)

        img = Image.fromarray(example, "RGB")

        img.save(cwd + str(i) + '_''Label_' + str(l) + '.jpg')

        print(example, l)
    coord.request_stop()
    coord.join(threads)
'''
train_path = './train.tfrecords'
test_path = './test.tfrecords'
train_whole_capacity = 20507
test_whole_capacity = 8779
class_num =29
train_batch_size = 32
test_batch_size = 32
img_size = 256

# flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)

#dropout probability
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)


#**----------------get data-----------------**#
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label':tf.FixedLenFeature([],tf.int64),
            'img_raw': tf.FixedLenFeature([],tf.string)
        }
    )
    img = tf.decode_raw(features['img_raw'],tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    #img = tf.cast(img, tf.float32)*(1./255.)-0.5

    label = tf.cast(features['label'],tf.int64)

    return img,label

train_X,train_y = read_and_decode(train_path)

train_X_batch,train_y_batch = tf.train.shuffle_batch(
    [train_X,train_y],
    batch_size= train_batch_size,
    capacity=train_whole_capacity,
    min_after_dequeue= 1000,
    num_threads=128

)

train_y_batch = tf.one_hot(train_y_batch,class_num,1,0)


test_X,test_y = read_and_decode(test_path)

test_X_batch,test_y_batch = tf.train.shuffle_batch(
    [test_X,test_y],
    batch_size= test_batch_size,
    capacity= test_whole_capacity,
    min_after_dequeue=100,
    num_threads=128
)

test_y_batch = tf.one_hot(test_y_batch,class_num,1,0)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
##**----------------定义函数-----------------**##
def weight(shape,stddev,w1,layer_name):
    var = tf.truncated_normal(shape,stddev)
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name = "weight_loss")  #L2范数正则化
        tf.add_to_collection("losses",weight_loss)  #将变量weight_loss放入losses变成一个列表
    return tf.Variable(var,name=layer_name)

def bias(shape,layer_name):
    init = tf.constant(0.0,shape=shape)
    return tf.Variable(init,name=layer_name)


def conv_layer(x,W,stride):
    return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')

def max_pooling(x,ksize,strides,padding='SAME'):
    return tf.nn.max_pool(x,ksize=ksize,strides=strides,padding=padding)



def batchnorm(Ylogits,is_test,iteration,offset,convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,iteration)
    bnepsilon = 1e-5
    if convolutional:
        mean,variance = tf.nn.moments(Ylogits,[0,1,2])

    else:
        mean,variance = tf.nn.moments(Ylogits,[0])


#**------------CNN-------------**#

sess = tf.Session()
path = './model.ckpt'
x = tf.placeholder(tf.float32,[None,img_size,img_size,3],name='image')
y = tf.placeholder(tf.float32,[None,class_num],name='label')


with tf.name_scope('Conv1'):
    w1 = weight([5,5,3,32],0.05,0.0,'w1')
    b1 = bias([32],'b1')
    #z1_BN = tf.matmul(x,w1)

    #batch_mean1,batch_var1 = tf.nn.moments(z1_BN,[0])
    stride = 1

    conv1 = tf.nn.conv2d(x,w1,strides=[1,stride,stride,1],padding='SAME')
    BN_mean1,BN_var1 = tf.nn.moments(conv1,[0,1,2],keep_dims=True)

    shift_1 = tf.Variable(tf.zeros([32]))
    scale_1 = tf.Variable(tf.zeros([32]))

    epsilon_1 = 1e-3

    BN_OUT_1 = tf.nn.batch_normalization(conv1,BN_mean1,BN_var1,shift_1,scale_1,epsilon_1)


    with tf.name_scope('h_conv1'):
        h_conv1 = tf.nn.relu(BN_OUT_1)
        #norm1 = tf.nn.lrn(h_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

with tf.name_scope('Pool1'):
    h_pool1 = max_pooling(h_conv1,ksize=[1,3,3,1],strides=[1,2,2,1])


with tf.name_scope('Conv2'):
    w2 = weight([5,5,32,32],0.05,0.0,'w2')
    b2 = bias([32],'b2')
    stride2 = 2
    with tf.name_scope('h_conv2'):
        #h_conv2 = tf.nn.relu(conv_layer(h_pool1,w2,stride2)+b2)
        #norm2 = tf.nn.lrn(h_conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
        conv2 = tf.nn.conv2d(h_pool1,w2,[1,stride2,stride2,1],padding='SAME')+b2

        BN_mean2,BN_var2 = tf.nn.moments(conv2,[0,1,2],keep_dims=True)
        shift_2 = tf.Variable(tf.zeros([32]))
        scale_2 = tf.Variable(tf.zeros([32]))

        epsilon_2 = 1e-3

        BN_OUT_2 = tf.nn.batch_normalization(conv2, BN_mean2, BN_var2, shift_2, scale_2, epsilon_2)

        h_conv2 = tf.nn.relu(BN_OUT_2)


with tf.name_scope('Pool2'):
    h_pool2 = max_pooling(h_conv2,ksize=[1,3,3,1],strides=[1,2,2,1])

with tf.name_scope('Conv3'):
    w3 = weight([5,5,32,64],0.05,0.0,'w3')
    b3 = bias([64],'b3')

    with tf.name_scope('h_conv3'):
        #h_conv3 = tf.nn.relu(conv_layer(h_pool2,w3,stride)+b3)
        #norm3 = tf.nn.lrn(h_conv3,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

        conv3 = tf.nn.conv2d(h_pool2, w3, [1,stride,stride,1],padding='SAME') + b3

        BN_mean3, BN_var3 = tf.nn.moments(conv3, [0, 1, 2], keep_dims=True)
        shift_3 = tf.Variable(tf.zeros([64]))
        scale_3 = tf.Variable(tf.zeros([64]))

        epsilon_3 = 1e-3

        BN_OUT_3 = tf.nn.batch_normalization(conv3, BN_mean3, BN_var3, shift_3, scale_3, epsilon_3)

        h_conv3 = tf.nn.relu(BN_OUT_3)
with tf.name_scope('Pool3'):
    h_pool3 = max_pooling(h_conv3,ksize=[1,3,3,1],strides=[1,4,4,1])




with tf.name_scope('Fc1'):
    fc_input = tf.reshape(h_pool3,[train_batch_size,-1])
    #dim = fc_input.get_shape()[1].value
    w_fc1 = weight([4*4*64,64],0.04,0.004,'w_fc1')
    b_fc1 = bias([64],'b_fc1')

    with tf.name_scope('h_fc1'):
        h_fc1 = tf.nn.relu(tf.matmul(fc_input,w_fc1)+b_fc1)

    keep_prob = tf.placeholder(tf.float32,name='my_keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob,name='my_h_fc1_drop')




with tf.name_scope('FC2'):

    w_fc2 = weight([64,class_num],0.1,0.0,'w_fc2')

    b_fc2 = bias([class_num],'b_fc2')

    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(tf.matmul(h_fc1,w_fc2)+b_fc2)






with tf.name_scope('Cross_Entropy'):
    #cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y,tf.int32),logits=tf.cast(prediction,tf.float32))
    #                               ,name='loss')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('cross_entropy',cross_entropy)


with tf.name_scope('Train_step'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,name='train_step')



correct_prediction = tf.equal(tf.arg_max(prediction,1),tf.arg_max(y,1))



##准确率
pre_num = tf.argmax(prediction,1,output_type='int32',name='output')
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


tf.summary.scalar('accuracy',accuracy)


merged = tf.summary.merge_all()


with tf.Session() as sess:

    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())
    #sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    train_writer = tf.summary.FileWriter('./model2',tf.get_default_graph())


    #coord = tf.train.Coordinator()

    #threads = tf.train.start_queue_runners(coord=coord,sess=sess)

    saver = tf.train.Saver()

    max_acc = 0

    for i in range(10001):

        img_xs,label_xs = sess.run([train_X_batch,train_y_batch])
        #print(label_xs)
        #label_xs = sess.run([train_y_batch])
        #print(img_xs.shape)
        #prediction = sess.run(prediction, feed_dict={x: img_xs, y: label_xs})
        #print(prediction)
        sess.run(train_step,feed_dict={x:img_xs,y:label_xs,keep_prob:0.75})


        if(i%100==0):
            print("epoch  {}".format(i))

            img_test_xs,label_test_xs = sess.run([test_X_batch,test_y_batch])


            #prediction = sess.run(correct_prediction,feed_dict={x:img_test_xs,y:label_test_xs})
            prediction, loss = sess.run([correct_prediction, cross_entropy],
                                        feed_dict={x: img_test_xs, y: label_test_xs})
            print('prediction:',prediction)

            acc = sess.run(accuracy,feed_dict={x:img_test_xs,y:label_test_xs,keep_prob:1.0})

            print("step = "+str(i)+" accuracy "+str(acc)+" loss :"+str(loss))



            summary = sess.run(merged,feed_dict={x:img_test_xs,y:label_test_xs,keep_prob:1})

            train_writer.add_summary(summary,i)


            if i==1000:
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
                with tf.gfile.FastGFile(path + '/' + 'model1000.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())


            elif i==2000:
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
                with tf.gfile.FastGFile(path + '/' + 'model2000.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

            elif i == 5000:
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
                with tf.gfile.FastGFile(path + '/' + 'model5000.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                break
            #if acc>0.996:
            #    break

    train_writer.close()


    coord.request_stop()
    coord.join(threads)

    sess.close()

print("训练结束")
