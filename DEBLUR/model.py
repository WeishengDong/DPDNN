from __future__ import division
import tensorflow.contrib.slim as slim
import numpy as np
import os
import time
import scipy.signal
import scipy.io
import h5py
from ops import *
from utils import *
from utils import pp
from bicubic_interp import bicubic_interp_2d
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages


TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))
class ReCNN(object):
    def __init__(self,sess, config, batch_size=64,is_train=True, checkpoint_dir=None):
        self.config = config
        self.sess = sess
        self.batch = batch_size
        self.is_train = is_train
        self.checkpoint_dir = checkpoint_dir

        # self.psf = self.prepare_psf(key = 'psf')
        self.psf = self.prepare_kernal(key='k1')




####### The single GPU model
        if is_train:
            self.test_Inputs1, self.test_Labels1, self.test_Inputs2, self.test_Labels2, self.test_Inputs3, self.test_Labels3,\
            self.test_Inputs4, self.test_Labels4 = self.prepare_data('Set10')
        else:
            self.test_Inputs1, self.test_Labels1 = self.prepare_data_test('Set10_test')

        # the Single_GPU model placeholder

        self.input = tf.placeholder(tf.float32,[batch_size,
                                                 self.test_Inputs1.shape[1],
                                                 self.test_Inputs1.shape[2],
                                                 1], name='noise_image')
        self.label = tf.placeholder(tf.float32,[batch_size,
                                                 self.test_Labels1.shape[1],
                                                 self.test_Labels1.shape[2],
                                                 1], name='denoise_image')

        # the Multi_GPU model placeholder
        self.lr = tf.placeholder(tf.float32,[], name='learning_rate')
        self.input1 = tf.placeholder(tf.float32,[batch_size,
                                                 self.test_Inputs1.shape[0],
                                                 self.test_Inputs1.shape[1],
                                                 1], name='noise_image1')
        self.label1 = tf.placeholder(tf.float32,[batch_size,
                                                 self.test_Labels1.shape[0],
                                                 self.test_Labels1.shape[1],
                                                 1], name='denoise_image1')
        self.input2 = tf.placeholder(tf.float32,[batch_size,
                                                 self.test_Inputs1.shape[0],
                                                 self.test_Inputs1.shape[1],
                                                 1], name='noise_image2')
        self.label2 = tf.placeholder(tf.float32,[batch_size,
                                                 self.test_Labels1.shape[0],
                                                 self.test_Labels1.shape[1],
                                                 1], name='denoise_image2')
        self.input3 = tf.placeholder(tf.float32,[batch_size,
                                                 self.test_Inputs1.shape[0],
                                                 self.test_Inputs1.shape[1],
                                                 1], name='noise_image3')
        self.label3 = tf.placeholder(tf.float32,[batch_size,
                                                 self.test_Labels1.shape[0],
                                                 self.test_Labels1.shape[1],
                                                 1], name='denoise_image3')
        self.input4 = tf.placeholder(tf.float32,[batch_size,
                                                 self.test_Inputs1.shape[0],
                                                 self.test_Inputs1.shape[1],
                                                 1], name='noise_image4')
        self.label4 = tf.placeholder(tf.float32,[batch_size,
                                                 self.test_Labels1.shape[0],
                                                 self.test_Labels1.shape[1],
                                                 1], name='denoise_image4')
    ##################### build the model ##########################################
        if is_train:
            ############ Multi_GPU model
            print("Defining multiple gpumodel and init the training operation...")
            self.train_op, self.multi_loss ,self.learning_rate = self.multi_gpu_model(config, self.lr, self.input1, self.label1, self.input2, self.label2, self.input3, self.label3, self.input4, self.label4)
            self.avg_loss = tf.add_n(self.multi_loss, name='avg_loss')/config.num_gpus
        else:
            ############ Single_GPU model
            self.output, self.avg_loss = self.get_loss(self.input, self.label, 1)
        #####~~~~~~~~~~~~~~~~~~~###########~~~~~~~~~~~~~~~~~~~~~~~~~~~###
        filter_data = dict()
        self.check_data_dic = filter_data
        self.train_all_writer  = tf.summary.FileWriter("./logs/train_all", sess.graph)
        self.train_writer = tf.summary.FileWriter("./logs/train")
        self.val_writer = tf.summary.FileWriter("./logs/val")

        if is_train:
            self.rate = tf.summary.scalar("Learning_rate", self.learning_rate)
        self.loss_sum = tf.summary.scalar("Loss_value", self.avg_loss)
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars]
        self.saver = tf.train.Saver(self.g_vars, max_to_keep=None)

        # self.var_to_restore = [val for val in t_vars if 'recon_1'and 'blur' in val.name]
        # self.saver1 = tf.train.Saver(self.var_to_restore,max_to_keep=None)

        # self.merged = tf.summary.merge_all()
    ##################### finish building the model##########################################

############################### MODEL ###############################################################################
    def multi_gpu_model(self, config, lr_ph, x_ph1, y_ph1, x_ph2, y_ph2, x_ph3, y_ph3, x_ph4, y_ph4):
        grads = []
        total_loss = []
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        optimizer = tf.train.AdamOptimizer(lr_ph)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(config.num_gpus):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("GPU_%d" % i)as scope:
                        if i == 0:
                            _, loss = self.get_loss(x_ph1, y_ph1, scope)
                        elif i == 1:
                            _, loss = self.get_loss(x_ph2, y_ph2, scope)
                        elif i == 2:
                            _, loss = self.get_loss(x_ph3, y_ph3, scope)
                        else:
                            _, loss = self.get_loss(x_ph4, y_ph4, scope)

                        total_loss.append(loss)
                        tf.get_variable_scope().reuse_variables()
                        grad_and_var = optimizer.compute_gradients(loss)
                        grads.append(grad_and_var)
        with tf.device("cpu:3"):
            # calculate the average gradients, and output to the TensorBoard
            averaged_gradients = self.average_gradients(grads)
            apply_gradient_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)
        return apply_gradient_op, total_loss,lr_ph

    def build_graph(self, _input):
        y = _input
        x = slim.conv2d_transpose(_input, 1, [19, 19], weights_initializer=tf.constant_initializer(self.psf),activation_fn=None, scope='initial')

        with tf.variable_scope("bloclk"):  # tf.AUTO_REUSE

            encode0, down0 = self.Encoding_block(x, name='Encoding_0')
            encode1, down1 = self.Encoding_block(down0, name='Encoding_1')
            encode2, down2 = self.Encoding_block(down1, name='Encoding_2')
            encode3, down3 = self.Encoding_block(down2, name='Encoding_3')

            media_end = self.Encoding_block_end(down3, name='Encoding_end')

            decode3 = self.Decoding_block(media_end, encode3, name='Decoding_3')
            decode2 = self.Decoding_block(decode3, encode2, name='Decoding_2')
            decode1 = self.Decoding_block(decode2, encode1, name='Decoding_1')
            decode0 = self.Decoding_block(decode1, encode0, name='Decoding_0')

            decoding_end = self.feature_decoding_end(decode0, name='end')
            conv_out = x + decoding_end

        x = self.recon(conv_out, x, y,  name='recon_1')

        for i in xrange(5):
            with tf.variable_scope("bloclk", reuse= True):#tf.AUTO_REUSE
                encode0, down0 = self.Encoding_block(x, name='Encoding_0')
                encode1, down1 = self.Encoding_block(down0, name='Encoding_1')
                encode2, down2 = self.Encoding_block(down1, name='Encoding_2')
                encode3, down3 = self.Encoding_block(down2, name='Encoding_3')

                media_end = self.Encoding_block_end(down3, name='Encoding_end')

                decode3 = self.Decoding_block(media_end, encode3, name='Decoding_3')
                decode2 = self.Decoding_block(decode3, encode2, name='Decoding_2')
                decode1 = self.Decoding_block(decode2, encode1, name='Decoding_1')
                decode0 = self.Decoding_block(decode1, encode0, name='Decoding_0')

                decoding_end  = self.feature_decoding_end(decode0, name= 'end')
                conv_out = x + decoding_end


            x = self.recon(conv_out, x, y, name = 'recon_%d'%(i+2))

        return x

    def get_loss(self, input_ph, label_ph, scope=1):
        y_hat = self.build_graph(input_ph)
        loss = tf.reduce_mean(tf.square(label_ph - y_hat))
        return y_hat, loss
        ################################################### function #######################################

    def recon(self, features, _recon, _noise, name):
        with tf.variable_scope(name):
            delta = tf.get_variable(name='delta', shape = [1], initializer=tf.constant_initializer(0.1))
            eta = tf.get_variable(name='eta', shape = [1],initializer=tf.constant_initializer(0.9))

            # blur_recon = slim.conv2d(_recon, 1, [19, 19], activation_fn=None, scope='blur')

            blur_recon = slim.conv2d(_recon, 1, [19, 19], weights_initializer= tf.constant_initializer(self.psf), activation_fn=None, scope='blur')
            err1 = slim.conv2d_transpose((blur_recon-_noise), 1, [3, 3],activation_fn=None, scope='deblur')
            err2 = _recon - features
            out = _recon - delta * (err1 + eta * err2)

        return out

    def Encoding_block(self, _input, name):
        with tf.variable_scope(name):
            # f_e = self.feature_encoding(_input, name='feature_extraction')
            conv = slim.conv2d(_input, 64, [3, 3], activation_fn=tf.nn.relu, scope='f_c_1')
            conv = slim.conv2d(conv, 64, [3, 3], activation_fn=tf.nn.relu, scope='f_c_2')
            f_e = slim.conv2d(conv, 64, [3, 3], activation_fn=tf.nn.relu, scope='f_c_3')
            down = slim.conv2d(f_e, 64, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='down_sampling')
        return f_e, down

    def Encoding_block_end(self, _input, name):
        with tf.variable_scope(name):
            f_e = slim.conv2d(_input, 64, [3, 3], activation_fn=tf.nn.relu, scope='feature_extraction_end')
        return f_e

    def Decoding_block(self, _input, map, name):
        with tf.variable_scope(name):
            up = self.up_sampling(_input, map, kernel_size=3, out_features=64, name='upsampling')
            cat = tf.concat([up, map], axis=3)
            cat = slim.conv2d(cat, 64, [1, 1], activation_fn=tf.nn.relu, scope='down_channal')
            # decoding = self.feature_decoding(cat, name='feature_decoding')
            conv = slim.conv2d(cat, 64, [3, 3], activation_fn=tf.nn.relu, scope='f_dc_1')
            conv = slim.conv2d(conv, 64, [3, 3], activation_fn=tf.nn.relu, scope='f_dc_2')
            conv = slim.conv2d(conv, 64, [3, 3], activation_fn=tf.nn.relu, scope='f_dc_3')
            decoding = slim.conv2d(conv, 64, [3, 3], activation_fn=tf.nn.relu, scope='f_dc_4')
            # decoding = slim.conv2d(conv, 64, [3, 3], activation_fn=tf.nn.relu, scope='f_dc_3')

        return decoding

    def feature_decoding_end(self, _input, name):
        with tf.variable_scope(name):
            conv = slim.conv2d(_input, 1, [3, 3], activation_fn=None, scope='f_dc_4')
        return conv

    def up_sampling(self, _input, label, kernel_size, out_features, name):
        with tf.variable_scope(name):
            batch_size = self.batch
            _input_h = int(_input.get_shape()[1])
            label_h1 = int(label.get_shape()[1])
            label_h2 = int(label.get_shape()[2])

            in_features = int(_input.get_shape()[-1])
            kernel = self.weight_variable_msra([kernel_size, kernel_size, out_features, in_features],
                                               name='kernel')

            Deconv = tf.nn.conv2d_transpose(_input, kernel,
                                            output_shape=[batch_size, label_h1, label_h2, out_features],
                                            strides=[1, 2, 2, 1], padding='SAME')
        return Deconv

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())
        ############################## MODEL ###############################################################################

        ####################################### TRAIN ###########################################
    def train(self,config,prin):
        """Train netmodel"""
        print ("Initializing all variable...")

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.graph.finalize()

        counter = 1
        start_time = time.time()
        pp.pprint(prin)
        if self.load(self.checkpoint_dir, config.model_dir):
            print("[*] load success...")
        else:
            print("[*] load failed....")

        # if self.load1('some', config.model_dir):
        #     print("[*] load success...")
        # else:
        #     print("[*] load failed....")

    ############################################## load Train data ############################################
        print("[prepare loading train data...")
        train_Inputs1, train_Labels1, train_Inputs2, train_Labels2 , train_Inputs3, train_Labels3, train_Inputs4, train_Labels4 = self.prepare_data('train')#train  Set10
        train_size = train_Inputs1.shape[2]
        image_size = train_Inputs1.shape[1]
        batch_index = range(train_size)
        print("[INFO] the train dataset is: %s, image_size is %d * %d" % (str(train_size*config.num_gpus), image_size, image_size))

    ############################################# load validate data ############################################
        print("[prepare loading validate data...")
        self.val_Inputs1, self.val_Labels1, self.val_Inputs2, self.val_Labels2 , self.val_Inputs3, self.val_Labels3, self.val_Inputs4, self.val_Labels4 = self.prepare_data('validate')#validate  Set10
        validate_size  = self.val_Inputs1.shape[2]
        val_image_size = self.val_Inputs1.shape[1]
        val_batch_size = config.batch_size
        validate_index = range(validate_size)
        random.shuffle(validate_index)
        validate_idxs = validate_size // val_batch_size
        print("[INFO] the validate dataset is: %s, image_size is %d * %d" % (str(validate_size*config.num_gpus), val_image_size, val_image_size))

        trn_counter = 0
        LOSS_trn = 0
        learning_rate = config.learning_rate
        ###############################################################################################################################################################################################################
        for epoch in xrange(config.epoch):
            random.shuffle(batch_index)
            batch_idxs = min(train_size, config.train_size) // config.batch_size
            save_step = np.array(batch_idxs// 4).astype(int)

            if epoch % 3 == 0:
                learning_rate = config.learning_rate * 0.5 ** (epoch // 3)

            # if epoch ==3:
            #     learning_rate = config.learning_rate * 0.5
            # elif epoch ==7:
            #     learning_rate = config.learning_rate * 0.5**2
            #
            # elif epoch ==11:
            #     learning_rate = config.learning_rate * 0.5**3
            #
            # elif epoch == 16:
            #     learning_rate = config.learning_rate * 0.5 ** 3

            for idx in xrange(0, batch_idxs): #batch_idxs


  ############################################## Prepare Train data ############################################
                Train_batch_input1 = np.transpose(train_Inputs1[:, :, batch_index[idx*config.batch_size:(idx+1)*config.batch_size]],[2, 1, 0])
                Train_batch_input1 = np.reshape(Train_batch_input1, list(Train_batch_input1.shape) + [1])
                Train_batch_label1 = np.transpose(train_Labels1[:, :, batch_index[idx*config.batch_size:(idx+1)*config.batch_size]],[2, 1, 0])
                Train_batch_label1 = np.reshape(Train_batch_label1, list(Train_batch_label1.shape) + [1])
                Train_batch_input2 = np.transpose(train_Inputs2[:, :, batch_index[idx*config.batch_size:(idx+1)*config.batch_size]],[2, 1, 0])
                Train_batch_input2 = np.reshape(Train_batch_input2, list(Train_batch_input2.shape) + [1])
                Train_batch_label2 = np.transpose(train_Labels2[:, :, batch_index[idx*config.batch_size:(idx+1)*config.batch_size]],[2, 1, 0])
                Train_batch_label2 = np.reshape(Train_batch_label2, list(Train_batch_label2.shape) + [1])

                Train_batch_input3 = np.transpose(train_Inputs3[:, :, batch_index[idx*config.batch_size:(idx+1)*config.batch_size]],[2, 1, 0])
                Train_batch_input3 = np.reshape(Train_batch_input3, list(Train_batch_input3.shape) + [1])
                Train_batch_label3 = np.transpose(train_Labels3[:, :, batch_index[idx*config.batch_size:(idx+1)*config.batch_size]],[2, 1, 0])
                Train_batch_label3 = np.reshape(Train_batch_label3, list(Train_batch_label3.shape) + [1])
                Train_batch_input4 = np.transpose(train_Inputs4[:, :, batch_index[idx*config.batch_size:(idx+1)*config.batch_size]],[2, 1, 0])
                Train_batch_input4 = np.reshape(Train_batch_input4, list(Train_batch_input4.shape) + [1])
                Train_batch_label4 = np.transpose(train_Labels4[:, :, batch_index[idx*config.batch_size:(idx+1)*config.batch_size]],[2, 1, 0])
                Train_batch_label4 = np.reshape(Train_batch_label4, list(Train_batch_label4.shape) + [1])

                fd_trn = {self.lr: learning_rate,
                          self.input1: Train_batch_input1, self.label1: Train_batch_label1,
                          self.input2: Train_batch_input2, self.label2: Train_batch_label2,
                          self.input3: Train_batch_input3, self.label3: Train_batch_label3,
                          self.input4: Train_batch_input4, self.label4: Train_batch_label4}
                _ = self.sess.run(self.train_op, feed_dict=fd_trn)
                nmse_train = self.sess.run(self.avg_loss, feed_dict=fd_trn)
                rate = self.sess.run(self.learning_rate, feed_dict=fd_trn)
                # loss_train_all = self.sess.run(self.merged, feed_dict=fd_trn)
                # self.train_all_writer.add_summary(loss_train_all, counter)


                LOSS_trn = LOSS_trn + nmse_train
                trn_counter = trn_counter +1
                counter += 1

            ########################## The validate running ####################33
                # if counter % 5000 == 0:  # 5000
                if counter % (save_step) == 0:  # save_step

                    avg_LOSS_validate = 0
                    for val_idx in xrange(0, validate_idxs):#validate_idxs
                        Validate_batch_input1 = np.transpose(self.val_Inputs1[:, :,validate_index[ val_idx*val_batch_size:(val_idx+1) * val_batch_size]],[2, 1, 0])
                        Validate_batch_input1 = np.reshape(Validate_batch_input1,list(Validate_batch_input1.shape) + [1])
                        Validate_batch_label1 = np.transpose(self.val_Labels1[:, :,validate_index[ val_idx*val_batch_size:(val_idx+1) * val_batch_size]],[2, 1, 0])
                        Validate_batch_label1 = np.reshape(Validate_batch_label1,list(Validate_batch_label1.shape) + [1])
                        Validate_batch_input2 = np.transpose(self.val_Inputs2[:, :,validate_index[ val_idx*val_batch_size:(val_idx+1) * val_batch_size]],[2, 1, 0])
                        Validate_batch_input2 = np.reshape(Validate_batch_input2,list(Validate_batch_input2.shape) + [1])
                        Validate_batch_label2 = np.transpose(self.val_Labels2[:, :,validate_index[ val_idx*val_batch_size:(val_idx+1) * val_batch_size]],[2, 1, 0])
                        Validate_batch_label2 = np.reshape(Validate_batch_label2,list(Validate_batch_label2.shape) + [1])

                        Validate_batch_input3 = np.transpose(self.val_Inputs3[:, :,validate_index[ val_idx*val_batch_size:(val_idx+1) * val_batch_size]],[2, 1, 0])
                        Validate_batch_input3 = np.reshape(Validate_batch_input3,list(Validate_batch_input3.shape) + [1])
                        Validate_batch_label3 = np.transpose(self.val_Labels3[:, :,validate_index[ val_idx*val_batch_size:(val_idx+1) * val_batch_size]],[2, 1, 0])
                        Validate_batch_label3 = np.reshape(Validate_batch_label3,list(Validate_batch_label3.shape) + [1])
                        Validate_batch_input4 = np.transpose(self.val_Inputs4[:, :,validate_index[ val_idx*val_batch_size:(val_idx+1) * val_batch_size]],[2, 1, 0])
                        Validate_batch_input4 = np.reshape(Validate_batch_input4,list(Validate_batch_input4.shape) + [1])
                        Validate_batch_label4 = np.transpose(self.val_Labels4[:, :,validate_index[ val_idx*val_batch_size:(val_idx+1) * val_batch_size]],[2, 1, 0])
                        Validate_batch_label4 = np.reshape(Validate_batch_label4,list(Validate_batch_label4.shape) + [1])

                        fd_val = {self.lr: learning_rate,
                                  self.input1: Validate_batch_input1, self.label1: Validate_batch_label1,
                                  self.input2: Validate_batch_input2, self.label2: Validate_batch_label2,
                                  self.input3: Validate_batch_input3, self.label3: Validate_batch_label3,
                                  self.input4: Validate_batch_input4, self.label4: Validate_batch_label4}
                        # _ = self.sess.run(self.train_op, feed_dict=fd_val) # if train, can run the step
                        nmse_validate = self.sess.run(self.avg_loss, feed_dict=fd_val)
                        avg_LOSS_validate = avg_LOSS_validate + nmse_validate


                    # loss_train = self.sess.run(self.merged, feed_dict=fd_trn)
                    # self.train_writer.add_summary(loss_train, counter)
                    # loss_val = self.sess.run(self.merged, feed_dict=fd_val)
                    # self.val_writer.add_summary(loss_val, counter)

                    avg_LOSS_validate = avg_LOSS_validate / validate_idxs
                    avg_MSE_validate  = avg_LOSS_validate* np.square(255.0)
                    avg_PSNR_validate = 20.0 * np.log10(255.0 / np.sqrt(avg_MSE_validate))

                    avg_loss_trn = LOSS_trn /trn_counter
                    avg_MSE_trn  = avg_loss_trn* np.square(255.0)
                    avg_PSNR_trn = 20.0 * np.log10(255.0 / np.sqrt(avg_MSE_trn))
                    trn_counter = 0
                    LOSS_trn  =0
                    print("Epoch: [%3d] [%4d/%4d][%7d] time: %10.4f, lr: %1.8f PSNR_trn: %2.4f, PSNR_val: %2.4f, loss_trn: %.8f, loss_val: %.8f" % (epoch + 1, idx + 1, batch_idxs,counter,
                                                                                                                                        time.time() - start_time, rate,avg_PSNR_trn, avg_PSNR_validate, avg_loss_trn, avg_LOSS_validate))
                    # if counter % 5000 == 0:
                    self.save(config.checkpoint_dir, counter, config.model_dir)
                    # self.save1('some', counter, config.model_dir)
        # self.val_writer.close()
        # self.train_writer.close()
        # self.train_all_writer.close()


    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def prepare_psf(self, key = 'psf'):
        psf = h5py.File('./data/' +key +'.mat')
        psf = np.transpose(np.array(psf['psf']).astype(np.float32), [ 1, 0])
        psf = np.reshape(psf, [1]+list(psf.shape) + [1])
        return psf

    def prepare_kernal(self, key='k1'):
        k_path = './new_k1_2.55/data/' + key + '.mat'
        k = h5py.File(k_path)
        psf = np.transpose(np.array(k['psf']).astype(np.float32), [ 1, 0])
        psf = np.reshape(psf, [1]+list(psf.shape) + [1])
        return psf

    def prepare_data_test(self, key):
        """prepare data

         Args:
            key: 'train' or 'test'.
         Returns:
            data_input_np: the input of data, size is [patch, height(LR), width(LR), channels]
            data_label_np: the label of data, size is [patch, height(HR), width(HR), channels]
        """
        input_path = './set10_k1_2.55/' + key + '_input.mat'
        label_path = './set10_k1_2.55/' + key + '_label.mat'

        data_input = h5py.File(input_path)
        data_label = h5py.File(label_path)

        key_input = 'I_LR'
        # key_input = 'img_noise'

        data_input = np.transpose(np.array(data_input[key_input]).astype(np.float32), [2, 1, 0])
        data_input = np.reshape(data_input, list(data_input.shape) + [1])

        key_input = 'I_HR'
        # key_input = 'img_clear'

        data_label = np.transpose(np.array(data_label[key_input]).astype(np.float32), [2, 1, 0])
        data_label = np.reshape(data_label, list(data_label.shape) + [1])
        return data_input, data_label

    def prepare_data(self, key='Set10'):
        """prepare data

         Args:
            key: 'train' or 'test'.
         Returns:
            data_input_np: the input of data, size is [patch, height(LR), width(LR), channels]
            data_label_np: the label of data, size is [patch, height(HR), width(HR), channels]
        """
        if key == 'Set10_test':
            input_path = './new_k1_2.55/data/' + key + '_input.mat'
            label_path = './new_k1_2.55/data/' + key + '_label.mat'

            data_input = h5py.File(input_path)
            data_label = h5py.File(label_path)

            key_input = 'I_LR'
            # key_input = 'img_noise'

            data_input = np.transpose(np.array(data_input[key_input]).astype(np.float32), [2, 1, 0])
            data_input = np.reshape(data_input, list(data_input.shape) + [1])

            key_input = 'I_HR'
            # key_input = 'img_clear'

            data_label = np.transpose(np.array(data_label[key_input]).astype(np.float32), [2, 1, 0])
            data_label = np.reshape(data_label, list(data_label.shape) + [1])
            return data_input, data_label

        else:

            input_path = './new_k1_2.55/data/' + key + '_input.mat'
            label_path = './new_k1_2.55/data/' + key + '_label.mat'

            data_input = h5py.File(input_path)
            data_label = h5py.File(label_path)
            key_input = 'I_LR'
            # key_input = 'img_noise'

            data_input = np.array(data_input[key_input]).astype(np.float32)
            all_num = np.array((data_input.shape[2] // 4) * 4).astype(int)
            cut_num = np.array(all_num / 4).astype(int)
            data_input1 = data_input[:, :, 0:cut_num]
            data_input2 = data_input[:, :, cut_num:cut_num * 2]
            data_input3 = data_input[:, :, cut_num * 2:cut_num * 3]
            data_input4 = data_input[:, :, cut_num * 3:cut_num * 4]

            key_input = 'I_HR'
            # key_input = 'img_clear'

            data_label = np.array(data_label[key_input]).astype(np.float32)
            data_label1 = data_label[:, :, 0:cut_num]
            data_label2 = data_label[:, :, cut_num:cut_num * 2]
            data_label3 = data_label[:, :, cut_num * 2:cut_num * 3]
            data_label4 = data_label[:, :, cut_num * 3:cut_num * 4]
            return data_input1, data_label1, data_input2, data_label2, data_input3, data_label3, data_input4, data_label4



    def save(self, checkpoint_dir, step, model_dir):

        checkpoint_dir = os.path.join(checkpoint_dir,model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, "ReconCNN"), global_step=step)


    def load(self, checkpoint_dir,model_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

            reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(checkpoint_dir, ckpt_name))
            var_to_shape_map = reader.get_variable_to_shape_map()
            # Print tensor name and values
            i=0
            for key in var_to_shape_map:
                i = i+1
                print("[%3d]tensor_name: "% i, key)
                # print(reader.get_tensor(key))
            return True
        else:
            return False


    def test(self, config, prin):
        pp.pprint(prin)

        if self.load(self.checkpoint_dir, config.model_dir):
            print("[*] load success")
        else:
            print("[!] load failed.")

        test_out_lst = []
        batch_size = 1
        Idx = self.test_Inputs1.shape[0] // batch_size
        if self.test_Inputs1.shape[0] % batch_size != 0:
            Idx += 1

        print('INFO [Test] starting test...')
        start_time = time.clock()
        num =0
        PSNR_all = 0
        for idx in xrange(0, self.test_Inputs1.shape[0]):
             num = num+1
             tmp = min((idx + 1) * batch_size, self.test_Inputs1.shape[0])
             input = self.test_Inputs1[idx * batch_size:tmp]  #
             label = self.test_Labels1[idx * batch_size:tmp]

             de_img, loss = self.sess.run([self.output, self.avg_loss],
                                                          feed_dict={self.input: input,
                                                                     self.label: label})
             MSE = loss * np.square(255.0)
             PSNR = 20.0 * np.log10(255.0 / np.sqrt(MSE))
             test_out_lst.append(de_img)
             PSNR_all = PSNR_all + PSNR
             # save_images(de_img, [1, 1], './sample/imag_%d.png' % idx)
        #
        PSNR_avg = PSNR_all/num
        # PSNR_avg = PSNR

        print("[INFO] [test] test finished, PSNR is %2.4f, cost time is: %.4f" % (PSNR_avg, (time.clock() - start_time)))
        # scipy.io.savemat('./data/result.mat', {'result': test_out_lst})
