"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def im2patch(im, step=1, win=64):
    patch_lst = []
    return patch_lst

def dore_size(x, shape):
    """transform [-1~1] to [0~255], and imresize
    """
    x = (x+1.)*127.5
    y = scipy.misc.imresize(x, shape, interp='bicubic')   # scipy.misc.imresize will transform number to 0~255

    return y


def get_image(image_path, scale, is_crop=True):
    """get image from image_path
    """
    img = scipy.misc.imread(image_path).astype(np.float)
    img_corp = crop_image(img, scale, is_crop)
    img_corp_trans = transform(img_corp)  # transform [0~255]->[-1~1]
    return img_corp_trans


def crop_image(img, scale, is_corp=True):
    """crop the image
    """
    # img = img[0:64, 0:64, :]
    img = center_crop(img, 64, 64)
    img_size = img.shape
    img_size_lst = list(img_size[0:2])
    img_size_new = tuple([x - x % scale for x in img_size_lst])
    if len(img_size) > 2:
        crop_img = img[0:img_size_new[0], 0:img_size_new[1], :]
    else:
        crop_img = img[0:img_size_new[0], 0:img_size_new[1]]
    return crop_img


def cal_psnr(clean_img, img, row, col):
    """compute the psnr
    """
    clean_img = np.array(clean_img).astype(np.float32)
    img = np.array(img).astype(np.float32)
    img_size = img.shape
    h = img_size[0]
    w = img_size[1]
    err = clean_img - img
    if len(img_size) > 2:
        err = err[0+row:h-row, 0+col:w-col, :]
    else:
        err = err[0+row:h-row, 0+col:w-col]
    err_vec = err.flatten()
    mse = np.mean(err_vec*err_vec)
    psnr = 10*math.log10(255**2/mse)
    return psnr


def save_images(images, size, image_path):
    """save images
    """
    num_im = size[0] * size[1]
    images_inverse_trans = inverse_transform(images[0:num_im])  # [-1~1]->[0~255]
    sz = images_inverse_trans.shape
    images_inverse_trans = np.reshape(images_inverse_trans[:, :, :, 0], [sz[0], sz[1], sz[2], 1])
    scipy.misc.imsave(image_path, merge(images_inverse_trans, size))

    #return imsave(inverse_transform(images[:num_im]), size, image_path)


def merge(images, size):
    """merge images
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    img = img.reshape((img.shape[0],img.shape[1]))
    return img

    # img = np.zeros((h * size[0], w * size[1]))
    # for idx, image in enumerate(images):
    #     i = idx % size[1]
    #     j = idx // size[1]
    #     img[j*h:j*h+h, i*w:i*w+w] = image
    # return img


def center_crop(image, crop_h, crop_w=None):
    """crop the center of image
    """
    if crop_w is None:
        crop_w = crop_h
    h, w = image.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    out = image[j:j+crop_h, i:i+crop_w]
    return out


def transform(image):
    """transform [0~255]->[-1,1]
    """
    return np.array(image)/127.5 - 1.


def inverse_transform(image):
    """transform
     when [-1~1]->[0,255]
    """
    image_max = np.max(image.flatten())
    if image_max > 1:
        return np.array(image)
    else:
        return np.array((image+1.)*127.5).astype(np.uint8)
        # return np.array(image * 255.0).astype(np.uint8)


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc", 
                        "sy": 1, "sx": 1, 
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv", 
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))


def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)


def visualize(sess, dcgan, config, option):
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
