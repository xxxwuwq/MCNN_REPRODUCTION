#!/usr/bin/env python
# -*-coding:utf-8-*-
# @Author : Weiqun Wu
# @Time : 2018-11-23

import math
import random
import os
import cv2 as cv
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)


def fspecial(ksize, sigma):
    """
    Generates 2d Gaussian kernel
    :param ksize: an integer, represents the size of Gaussian kernel
    :param sigma: a float, represents standard variance of Gaussian kernel
    :return: 2d Gaussian kernel, the shape is [ksize, ksize]
    """
    # [left, right)
    left = -ksize / 2 + 0.5
    right = ksize / 2 + 0.5
    x, y = np.mgrid[left:right, left:right]
    # generate 2d Gaussian Kernel by normalization
    gaussian_kernel = np.exp(-(np.square(x) + np.square(y)) / (2 * np.power(sigma, 2))) / (2 * np.power(sigma, 2)).sum()
    sum = gaussian_kernel.sum()
    normalized_gaussian_kernel = gaussian_kernel / sum

    return normalized_gaussian_kernel


def get_avg_distance(position, points, k):
    """
    Computes the average distance between a pedestrian and its k nearest neighbors
    :param position: the position of the current point, the shape is [1,1]
    :param points: the set of all points, the shape is [num, 2]
    :param k: a integer, represents the number of mearest neibor we want
    :return: the average distance between a pedestrian and its k nearest neighbors
    """

    # in case that only itself or the k is lesser than or equal to num
    num = len(points)
    if num == 1:
        return 1.0
    elif num <= k:
        k = num - 1

    euclidean_distance = np.zeros((num, 1))
    for i in range(num):
        x = points[i, 1]
        y = points[i, 0]
        # Euclidean distance
        euclidean_distance[i, 0] = math.sqrt(math.pow(position[1] - x, 2) + math.pow(position[0] - y, 2))

    # the all distance between current point and other points
    euclidean_distance[:, 0] = np.sort(euclidean_distance[:, 0])
    avg_distance = euclidean_distance[1:k + 1, 0].sum() / k
    return avg_distance


def get_density_map(scaled_crowd_img_size, scaled_points, knn_phase, k, scaled_min_head_size, scaled_max_head_size):
    """
    Generates the correspoding ground truth density map
    :param scaled_crowd_img_size: the size of ground truth density map
    :param scaled_points: the position set of all points, but were divided into scale already
    :param knn_phase: True or False, determine wheather use geometry-adaptive Gaussian kernel or general one
    :param k: number of k nearest neighbors
    :param scaled_min_head_size: the scaled maximum value of head size for original pedestrian head
                          (in corresponding density map should be divided into scale)
    :param scaled_max_head_size:the scaled minimum value of head size for original pedestrian head
                          (in corresponding density map should be divided into scale)
    :return: density map, the shape is [scaled_img_size[0], scaled_img_size[1]]
    """

    h, w = scaled_crowd_img_size[0], scaled_crowd_img_size[1]

    density_map = np.zeros((h, w))
    # In case that there is no one in the image
    num = len(scaled_points)
    if num == 0:
        return density_map
    for i in range(num):
        # For a specific point in original points label of dataset, it represents as position[oy, ox],
        # so points[i, 1] is x, and points[i, 0] is y Also in case that the negative value
        x = min(h, max(0, abs(int(math.floor(scaled_points[i, 1])))))
        y = min(w, max(0, abs(int(math.floor(scaled_points[i, 0])))))
        # now for a specific point, it represents as position[x, y]

        position = [x, y]

        sigma = 1.5
        beta = 0.3
        ksize = 25
        if knn_phase:
            avg_distance = get_avg_distance(position, scaled_points, k=k)
            avg_distance = max(min(avg_distance, scaled_max_head_size), scaled_min_head_size)
            sigma = beta * avg_distance
            ksize = 1.0 * avg_distance

        # Edge processing
        x1 = x - int(math.floor(ksize / 2))
        y1 = y - int(math.floor(ksize / 2))
        x2 = x + int(math.ceil(ksize / 2))
        y2 = y + int(math.ceil(ksize / 2))

        if x1 < 0 or y1 < 0 or x2 > h or y2 > w:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(h, x2)
            y2 = min(w, y2)

            tmp = x2 - x1 if (x2 - x1) < (y2 - y1) else y2 - y1
            ksize = min(tmp, ksize)

        ksize = int(math.floor(ksize / 2))
        H = fspecial(ksize, sigma)
        density_map[x1:x1 + ksize, y1:y1 + ksize] = density_map[x1:x1 + ksize, y1:y1 + ksize] + H
    return np.asarray(density_map)


def get_cropped_crowd_image(ori_crowd_img, points, crop_size):
    """
    Crops a sub-crowd image randomly
    :param ori_crowd_img: original crowd image, the shape is [h, w, channel]
    :param points: the original position set of all points
    :param crop_size: the cropped crowd image size we need
    :return: cropped crowd image, cropped points, cropped crowd count
    """
    h, w = ori_crowd_img.shape[0], ori_crowd_img.shape[1]

    # if the original image size < the crooped_image size, reduce the crop size
    if h < crop_size or w < crop_size:
        crop_size = crop_size // 2

    # random to get the crop area
    x1 = random.randint(0, h - crop_size)
    y1 = random.randint(0, w - crop_size)
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # the crowd image after croppig
    cropped_crowd_img = ori_crowd_img[x1:x2, y1:y2, ...]

    # img_gray_crop = img_gray[x1:x2, y1:y2]
    # img_gray_crop = cv.resize(img_gray_crop, (img_gray_crop.shape[1] // (scale), img_gray_crop.shape[0] // (scale)))

    # Computes the points after cropping
    cropped_points = []
    for i in range(len(points)):
        if x1 <= points[i, 1] <= x2 and y1 <= points[i, 0] <= y2:
            points[i, 0] = points[i, 0] - y1
            points[i, 1] = points[i, 1] - x1
            cropped_points.append(points[i])
    cropped_points = np.asarray(cropped_points)
    cropped_crowd_count = len(cropped_points)
    return cropped_crowd_img, cropped_points, cropped_crowd_count


def get_scaled_crowd_image_and_points(crowd_img, points, scale):
    """
    Gets scaled crowc image and scaled points for corresponding density map
    :param crowd_image: the crowd image that wanted to be scaled to generate ground truth density map
    :param points: the position set of all points that wanted to be scaled to generate ground truth density map
    :param scale: the scale factor
    :return: sacled crowd image, scaled points
    """
    h = crowd_img.shape[0]
    w = crowd_img.shape[1]
    scaled_crowd_img = cv.resize(crowd_img, (w // scale, h // scale))
    for i in range(len(points)):
        points[i] = points[i] / scale

    return scaled_crowd_img, points


def read_train_data(img_path, gt_path, crop_size=256, scale=8, knn_phase=True, k=2, min_head_size=16, max_head_size=200):
    """
    read_the trianing data from datasets ad the input and label of network
    :param img_path: the crowd image path
    :param gt_path: the label(ground truth) data path
    :param crop_size: the crop size
    :param scale: the scale factor, accorting to the accumulated downsampling factor
    :param knn_phase: True or False, determines wheather to use geometry-adaptive Gaussain kernel or general one
    :param k:  a integer, the number of neareat neighbor
    :param min_head_size: the minimum value of the head size in original crowd image
    :param max_head_size: the maximum value of the head size in original crowd image
    :return: the crwod image as the input of network, the scaled density map as the ground truth of network,
             the ground truth crowd count
    """

    ori_crowd_img = cv.imread(img_path)
    # read the .mat file in dataset
    label_data = loadmat(gt_path)
    points = label_data['image_info'][0][0]['location'][0][0]
    # crowd_count = label_data['image_info'][0][0]['number'][0][0]
    cropped_crowd_img, cropped_points, cropped_crowd_count = get_cropped_crowd_image(ori_crowd_img, points, crop_size=crop_size)

    cropped_scaled_crowd_img, cropped_scaled_points = get_scaled_crowd_image_and_points(cropped_crowd_img, cropped_points, scale=scale)
    # cropped_scaled_crowd_count = cropped_crowd_count
    cropped_scaled_crowd_img_size = [cropped_scaled_crowd_img.shape[0], cropped_scaled_crowd_img.shape[1]]
    scaled_min_head_size = min_head_size / scale
    scaled_max_head_size = max_head_size / scale

    # after cropped and scaled
    density_map = get_density_map(cropped_scaled_crowd_img_size, cropped_scaled_points,
                                  knn_phase, k, scaled_min_head_size, scaled_max_head_size)

    # cropped_crowd_img = np.asarray(cropped_crowd_img)
    cropped_crowd_img = cropped_crowd_img.reshape((1, cropped_crowd_img.shape[0], cropped_crowd_img.shape[1], cropped_crowd_img.shape[2]))
    cropped_crowd_count = np.asarray(cropped_crowd_count).reshape((1, 1))
    cropped_scaled_density_map = density_map.reshape((1, density_map.shape[0], density_map.shape[1], 1))

    return cropped_crowd_img, cropped_scaled_density_map, cropped_crowd_count


def read_test_data(img_path, gt_path, scale=8, deconv_is_used=False, knn_phase=True, k=2, min_head_size=16, max_head_size=200):
    """
    read_the testing data from datasets ad the input and label of network
    :param img_path: the crowd image path
    :param gt_path: the label(ground truth) data path
    :param scale: the scale factor, accorting to the accumulated downsampling factor
    :param knn_phase: True or False, determines wheather to use geometry-adaptive Gaussain kernel or general one
    :param k:  a integer, the number of neareat neighbor
    :param min_head_size: the minimum value of the head size in original crowd image
    :param max_head_size: the maximum value of the head size in original crowd image
    :return: the crwod image as the input of network, the scaled density map as the ground truth of network,
             the ground truth crowd count
    """

    ori_crowd_img = cv.imread(img_path)

    # read the .mat file in dataset
    label_data = loadmat(gt_path)
    points = label_data['image_info'][0][0]['location'][0][0]
    crowd_count = label_data['image_info'][0][0]['number'][0][0]
    h, w = ori_crowd_img.shape[0], ori_crowd_img.shape[1]

    if deconv_is_used:
        h_ = h - (h // scale) % 2
        rh = h_ / h
        w_ = w - (w // scale) % 2
        rw = w_ / w
        ori_crowd_img = cv.resize(ori_crowd_img, (w_, h_))
        points[:, 1] = points[:, 1] * rh
        points[:, 0] = points[:, 0] * rw
    # scaled_crowd_img, scaled_points = ori_crowd_img,points
    scaled_crowd_img, scaled_points = get_scaled_crowd_image_and_points(ori_crowd_img, points, scale=scale)
    # scaled_crowd_count = crowd_count
    scaled_crowd_img_size = [scaled_crowd_img.shape[0], scaled_crowd_img.shape[1]]
    scaled_min_head_size = min_head_size / scale
    scaled_max_head_size = max_head_size / scale

    # after cropped and scaled
    density_map = get_density_map(scaled_crowd_img_size, scaled_points, knn_phase, k, scaled_min_head_size, scaled_max_head_size)
    ori_crowd_img = ori_crowd_img.reshape((1, ori_crowd_img.shape[0], ori_crowd_img.shape[1], ori_crowd_img.shape[2]))
    crowd_count = np.asarray(crowd_count).reshape((1, 1))
    scaled_density_map = density_map.reshape((1, density_map.shape[0], density_map.shape[1], 1))

    return ori_crowd_img, scaled_density_map, crowd_count


def mae_metric(ground_truth, inference):
    return np.abs(np.subtract(ground_truth, inference)).mean()


def se_metric(ground_truth, inference):
    return np.power(np.subtract(ground_truth, inference), 2).mean()


def show_density_map(density_map):
    """
    show the density map to help us analysis the distribution of the crowd
    :param density_map: the density map, the shape is [h, w]
    """

    plt.imshow(density_map, cmap='jet')
    plt.show()


def set_gpu(gpu=0):
    """
    the gpu used setting
    :param gpu: gpu id
    :return:
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


if __name__ == '__main__':
    crowd_img, density_map, cropped_crowd_count = read_test_data('./IMG_2.jpg', './GT_IMG_2.mat')
    # print(density_map[0, :, :, 0])
    # img_path = '../IMG_2.jpg'
    # ori_crowd_img = cv.imread(img_path)
    # gt_path = '../GT_IMG_2.mat'
    # label_data = loadmat(gt_path)
    # points = label_data['image_info'][0][0]['location'][0][0]
    #
    # dmp = get_density_map([ori_crowd_img.shape[0], ori_crowd_img.shape[1]], points, knn_phase=False, k=1, scaled_min_head_size=2, scaled_max_head_size=3)
    # print(dmp.shape)
    # dmp = cv.resize(dmp, (dmp.shape[1] // 8, dmp.shape[0] // 8))
    # print(dmp.shape)
    # show_density_map(dmp)
    # show_density_map(ori_crowd_img[:, :, 0])
    sum = np.sum(np.sum(density_map))
    print(sum, cropped_crowd_count)
    show_density_map(density_map[0, :, :, 0])
    show_density_map(crowd_img[:, :, 0])
