import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string("path_to_test",'./test/img/alagilles-syndrome_0_3595.jpg',"direction to tested image")



def main(*args):
    path = FLAGS.path_to_test
    print(path)



if __name__ =="__main__":
    tf.app.run()