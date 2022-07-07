import os
import cv2
import numpy as np
import tensorflow as tf 
import network
import guided_filter
from tqdm import tqdm
import argparse


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image


def cartoonize(load_folder, save_folder, model_path):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    
    
    for root, dirs, files in tqdm(os.walk(load_folder, topdown=False), desc='Folders', total=len(os.listdir(load_folder))):
        if not os.path.exists(root.replace(load_folder, save_folder)):
            os.makedirs(root.replace(load_folder, save_folder), exist_ok=True)

        for name in tqdm(files, desc='Images', leave=False):
            load_path = os.path.join(root, name)
            save_path = load_path.replace(load_folder, save_folder)
            image = cv2.imread(load_path)
            image = resize_crop(image)
            batch_image = image.astype(np.float32)/127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = (np.squeeze(output)+1)*127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)
    
parser = argparse.ArgumentParser(description='Create ImageNet-Cartoon')
parser.add_argument('--model_path', type=str, default='saved_models', help="Location of the pre-trained models")
parser.add_argument('--load_folder', type=str, default='path/to/imagenet/val', help="Path to ImageNet val folder")
parser.add_argument('--save_folder', type=str, default='../datasets/imagenet-cartoon', help="Path where to save ImageNet-Cartoon") 
args = parser.parse_args()
    
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder, exist_ok=True)
cartoonize(args.load_folder, args.save_folder, args.model_path)
    

    