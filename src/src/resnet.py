import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.python.framework import graph_util
import os
train_path = './train.tfrecords'
test_path = './test.tfrecords'
train_whole_capacity = 20507
test_whole_capacity = 8779
num_class =29
train_batch_size = 32
test_batch_size = 32
img_size = 256
lr = tf.Variable(0.001, dtype=tf.float32)
path = './model3'
if not os.path.exists(path):
    os.mkdir(path)

sess = tf.Session()

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename],shuffle=True)

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
    label = tf.cast(features['label'],tf.int32)

    return img,label

train_X, train_y = read_and_decode(train_path)

train_X_batch, train_y_batch = tf.train.shuffle_batch(
        [train_X, train_y],
        batch_size=train_batch_size,
        capacity=train_whole_capacity,
        min_after_dequeue=100,
        num_threads=128

    )

train_y_batch = tf.one_hot(train_y_batch, num_class, 1, 0)

test_X, test_y = read_and_decode(test_path)

test_X_batch, test_y_batch = tf.train.shuffle_batch(
        [test_X, test_y],
        batch_size=test_batch_size,
        capacity=test_whole_capacity,
        min_after_dequeue=100,
        num_threads=128
    )

test_y_batch = tf.one_hot(test_y_batch, num_class, 1, 0)


x = tf.placeholder(tf.float32,[None,256,256,3],name='image')
y_ = tf.placeholder(tf.float32,[None,num_class],name='pred')

pred,end_point = resnet_v2.resnet_v2_50(x,num_classes=num_class,is_training=True)
pred = tf.reshape(pred,shape=[-1,num_class])


##定义损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=train_y_batch))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

##准确率
pre_num = tf.argmax(pred,1,output_type='int32',name='output')
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y_,1)),tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    #线程
    coord = tf.train.Coordinator()
    #启动QueueRunner(),使文件名进入队列
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    step =0

    while True:
        step+=1
        img_xs, label_xs = sess.run([train_X_batch, train_y_batch])

        _,loss_ = sess.run([optimizer,loss],feed_dict={x:img_xs,y_:label_xs})

        print("step {} ,train_loss :{}".format(step,loss_))
        if step %50==0:
            _loss,acc_train = sess.run([loss,accuracy],feed_dict={x:img_xs,y_:label_xs})

            print('-'*20)
            print('step: {} ,train_acc: {}, loss: {}'.format(step,acc_train,_loss))

            print('-'*20)

            if step==1500:
                saver.save(sess,save_path=path,global_step=step)
                saver.save(sess,"./model3/model.cpkt",global_step=step)
                constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def,["output"])
                with tf.gfile.FastGFile(path+'/'+'model1500.pb',mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
            elif step==2500:
                saver.save(sess,save_path=path,global_step=step)
                saver.save(sess, "./model3/model.cpkt", global_step=step)
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
                with tf.gfile.FastGFile(path+'/' + 'model2500.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

            elif step == 3500:
                saver.save(sess, save_path=path, global_step=step)
                saver.save(sess, "./model3/model.cpkt", global_step=step)
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
                with tf.gfile.FastGFile(path + '/' + 'model3500.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                break
    print("trainning end")
    coord.request_stop()
    coord.join(threads)




