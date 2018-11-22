#!/usr/bin/env python
# -*-coding:utf-8-*-
# @Author : Weiqun Wu
# @Time : 2018-11-23

import tensorflow as tf
import numpy as np
import os
import random
import time
from utils import *
from networks import multi_column_cnn
from configs import *

np.set_printoptions(threshold=np.inf)


def train():
    set_gpu(1)
    dataset = 'A'
    # training dataset
    img_root_dir = r'D:\weiqunWu\Datasets\ShanghaiTech\part_' + dataset + r'_final\train_data\images\\'
    gt_root_dir = r'D:\weiqunWu\Datasets\ShanghaiTech\part_' + dataset + r'_final\train_data\ground_truth\\'
    # testing dataset
    val_img_root_dir = r'D:\weiqunWu\Datasets\ShanghaiTech\part_' + dataset + r'_final\test_data\images\\'
    val_gt_root_dir = r'D:\weiqunWu\Datasets\ShanghaiTech\part_' + dataset + r'_final\test_data\ground_truth\\'

    # training dataset file list
    img_file_list = os.listdir(img_root_dir)
    gt_img_file_list = os.listdir(gt_root_dir)

    # testing dataset file list
    val_img_file_list = os.listdir(val_img_root_dir)
    val_gt_file_list = os.listdir(val_gt_root_dir)

    cfig = ConfigFactory()

    # place holder
    input_img_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    density_map_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 1))

    # network generation
    inference_density_map = multi_column_cnn(input_img_placeholder)

    # density map loss
    density_map_loss = 0.5 * tf.reduce_sum(tf.square(tf.subtract(density_map_placeholder, inference_density_map)))

    # jointly training
    joint_loss = density_map_loss
    # optimizer = tf.train.MomentumOptimizer(configs.learing_rate, momentum=configs.momentum).minimize(joint_loss)
    # adam optimizer
    optimizer = tf.train.AdamOptimizer(cfig.lr).minimize(joint_loss)

    init = tf.global_variables_initializer()
    # start session
    sess = tf.InteractiveSession()
    sess.run(init)

    file_path = cfig.log_router

    # training log route
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # model saver route
    if not os.path.exists(cfig.ckpt_router):
        os.makedirs(cfig.ckpt_router)
    log = open(cfig.log_router + cfig.name + r'_training.logs', mode='a+', encoding='utf-8')

    saver = tf.train.Saver(max_to_keep=cfig.max_ckpt_keep)
    ckpt = tf.train.get_checkpoint_state(cfig.ckpt_router)

    if ckpt and ckpt.model_checkpoint_path:
        print('load model')
        saver.restore(sess, ckpt.model_checkpoint_path)

    # start training
    for i in range(cfig.total_iters):
        # training
        for file_index in range(len(img_file_list)):
            img_path = img_root_dir + img_file_list[file_index]
            gt_path = gt_root_dir + 'GT_' + img_file_list[file_index].split(r'.')[0] + '.mat.'
            img, gt_dmp, gt_count = read_train_data(img_path, gt_path, scale=4)

            feed_dict = {input_img_placeholder: (img - 127.5) / 128, density_map_placeholder: gt_dmp}

            _, inf_count, loss = sess.run([optimizer, tf.reduce_sum(inference_density_map), joint_loss],
                                          feed_dict=feed_dict)
            format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            format_str = 'step %d, joint loss=%.5f, inference= %.5f, gt=%d'
            log_line = format_time, img_file_list[file_index], format_str % (i * len(img_file_list) + file_index, loss, inf_count, gt_count)
            log.writelines(str(log_line) + '\n')
            print(log_line)

        saver.save(sess, cfig.ckpt_router + '/v1', global_step=i)

        if i % 50 == 0:
            val_log = open(cfig.log_router + cfig.name + r'_validating_' + str(i) +  '_.logs', mode='w', encoding='utf-8')
            absolute_error = 0.0
            square_error = 0.0
            # validating
            for file_index in range(len(val_img_file_list)):
                img_path = val_img_root_dir + val_img_file_list[file_index]
                gt_path = val_gt_root_dir + 'GT_' + val_img_file_list[file_index].split(r'.')[0] + '.mat.'
                img, gt_dmp, gt_count = read_test_data(img_path, gt_path, scale=4)

                feed_dict = {input_img_placeholder: (img - 127.5) / 128, density_map_placeholder: gt_dmp}
                _, inf_count, loss = sess.run([optimizer, tf.reduce_sum(inference_density_map), joint_loss], feed_dict=feed_dict)

                format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                format_str = 'step %d, joint loss=%.5f, inference= %.5f, gt=%d'
                absolute_error = absolute_error + np.abs(np.subtract(gt_count, inf_count)).mean()
                square_error = square_error + np.power(np.subtract(gt_count, inf_count), 2).mean()
                log_line = format_time, val_img_file_list[file_index], format_str % (file_index, loss, inf_count, gt_count)
                val_log.writelines(str(log_line) + '\n')
                print(log_line)
            mae = absolute_error / len(val_img_file_list)
            rmse = np.sqrt(absolute_error / len(val_img_file_list))
            val_log.writelines(str('MAE_' + str(mae) + '_MSE_' + str(rmse)) + '\n')
            val_log.close()
            print(str('MAE_' +str(mae) + '_MSE_' + str(rmse)))


if __name__ == '__main__':
    train()
