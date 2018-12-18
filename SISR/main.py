import os
import numpy as np
import tensorflow as tf

from utils import pp
from model import ReCNN


os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

flags = tf.app.flags
flags.DEFINE_integer("epoch",500,"Epoch to train")
# flags.DEFINE_integer("batch_size",16, "the size of batch images")
flags.DEFINE_integer("batch_size",1, "the size of batch images")

flags.DEFINE_integer("train_size", np.inf, "the max size of train images")
# flags.DEFINE_integer("num_gpus",4,"The number of GPU")
flags.DEFINE_integer("num_gpus",1,"The number of GPU")


flags.DEFINE_float("learning_rate",0.0001,"Learning  rate for Adam")
# flags.DEFINE_boolean("is_train",True,"True for training, False for testing")
flags.DEFINE_boolean("is_train",False,"True for training, False for testing")
flags.DEFINE_string("checkpoint_dir","checkpoint", "directory name to save the checkpoint")
flags.DEFINE_string("model_dir","recnn_v3", "directory name to save the checkpoint")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(FLAGS.__flags)

    # pp.pprint(flags.FLAGS.__flags)
    if FLAGS.is_train:
        print("[*]Defining session...")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True,
                                      gpu_options=gpu_options)
        session = tf.Session(config=session_conf)
    else:
        session = tf.Session()

    with session as sess:
        recnn = ReCNN(sess, FLAGS, batch_size=FLAGS.batch_size, is_train=FLAGS.is_train,
                        checkpoint_dir=FLAGS.checkpoint_dir)

        if FLAGS.is_train:
            recnn.train(FLAGS, FLAGS.__flags)
        else:
            recnn.test(FLAGS, FLAGS.__flags)

if __name__ == '__main__':
    tf.app.run()