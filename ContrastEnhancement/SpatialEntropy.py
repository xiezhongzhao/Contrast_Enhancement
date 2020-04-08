#!/usr/bin/env python
# encoding: utf-8

# @Author: Xie Zhongzhao
# @Date  : 2019-12-02 10:12:00

import os
from os import path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

class QRCM(object):
    '''
    A quality-aware relative contrast measure (QRCM) is proposed in this paper.
    <SMI AND PAGERANK-BASED CONTRAST ENHANCEMENT AND QUALITY-AWARE RELATIVE CONTRAST MEASURE>
    This measure considers both the level of relative contrast enhancement between input and output images
    and distortions resulting from the enhancement process. The measure produces a number
    in the range [−1, 1] where -1 and 1 refer to full level of contrast degradation and improvement,
    respectively.
    '''
    def __int__(self):

        pass

    def conv2(self, x, y, mode='same'):
        return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

    def gradient(self, img):
        '''
        calculate the gradient magnitude map
        :param path:
        :return: gradient magnitude map
        '''
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = img[:, :, 2]

        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3

        bk = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        bk = bk / np.sum(bk)

        img_bk = self.conv2(img, bk, 'same')

        grad_x = self.conv2(img_bk, kernelx, 'same')
        grad_y = self.conv2(img_bk, kernely, 'same')

        grad_map = np.sqrt(np.square(grad_x) + np.square(grad_y))

        return grad_map

    def rcm(self, origImg, proImg):
        '''
        the relative contrast measure RCM
        :param orig_path: the original image
        :param pro_path: the processed image
        :return: RCM value
        '''
        EPS = 1e-6
        G_o = self.gradient(origImg)
        G_p = self.gradient(proImg)

        G_po = (G_p-G_o) / (G_p+G_o+EPS)

        w1 = G_o / sum(sum(G_o))

        RCM = np.sum(G_po * w1)

        return RCM

    def qvalue(self, origImg, proImg):
        '''
        the image quality
        :param orig_path: the original image path
        :param pro_path: the processed image path
        :return: Q value
        '''
        T = 255 / np.sqrt(2)
        G_o = self.gradient(origImg)
        G_p = self.gradient(proImg)

        m, n = (G_o.shape[0], G_p.shape[1])

        GMS = (2*G_o*G_p+T) / (np.square(G_o)+np.square(G_p)+T)
        mu = np.mean(GMS)
        w2 = 1 / (1+G_o)

        Q = 1 - (1/(m*n))*np.sum(np.abs(GMS-mu)*w2)

        return Q

    def qrcm(self, origImg, proImg):
        '''
        the quality-aware relative contrast measure
        :return: the QRCM value in the range of [-1,1]
        '''
        RCM = self.rcm(origImg, proImg)
        Q = self.qvalue(origImg, proImg)
        # print("RCM: ", RCM)
        # print("Q: ", Q)

        Qrcm = RCM*Q if RCM>=0 else (1+RCM)*Q - 1
        Qrcm = '{:.4f}'.format(Qrcm)

        return Qrcm

class SECEDCT(object):
    """
    The algorithm introduces a new method to compute the spatial entropy of pixels
    using spatial distribution of pixel gray levels. The algorithm is from the paper,
    Spatial Entropy-Based Global and Local Image Contrast Enhancement, proposed by Turgay Celik
    """
    def __init__(self, path_to_img):
        """
        :param path_to_img : full path to the image file
        """
        self._EPS = 1e-6
        self.yd = 0
        self.yu = 255
        self.img_bgr = cv2.imread(path_to_img, 1)


        self.height, self.width = self.img_bgr.shape[0:2] # avoid the odd error of DCT and IDCT 
        new_height = self.height
        new_width = self.width
        if self.height % 2 == 1:
            new_height = self.height + 1

        if self.width % 2 == 1:
            new_width = self.width + 1
        self.img_bgr = cv2.resize(self.img_bgr, (new_width, new_height), cv2.INTER_AREA)


        if self.img_bgr is None:
            raise Exception("cv2.imread failed! Please check if the path is valid")

        self.img_size = self.img_bgr.shape

        self.img_gray = cv2.cvtColor(
            self.img_bgr,
            cv2.COLOR_BGR2GRAY  # cv2 bgr2gray doing the same as NTSC
        )  # I in the paper

        self.img_hsv = cv2.cvtColor(
            self.img_bgr,
            cv2.COLOR_BGR2HSV  # cv2 bgr2hsv doing the same as NTSC
        )  # I in the paper

    def spatialHistgram(self, img_hsv):
        '''
        2D spatial histogram
        :return: 2D spatial histogram
        '''
        histogram = dict()

        # K = np.unique(img_hsv[:, :, 2])  # the distinct gray levels K
        K = [i for i in range(256)]
        img = img_hsv[:, :, 2]

        x, y = img.shape[0:2] # reduce the time by resizing the image
        img = cv2.resize(img, (int(y / 2), int(x / 2)))

        H = self.img_size[0]
        W = self.img_size[1]
        ratio = H / W  # the aspect ratio r = H/W
        k_num = len(K)

        M = np.rint((k_num * ratio) ** 0.5)  # 2D histogram is M*N
        N = np.rint((k_num / ratio) ** 0.5)  # the total number of the grids on

        region_list = list()
        for m in range(1, int(M) + 1):
            for n in range(1, int(N) + 1):
                left = int((m - 1) / M * H)
                right = int((m / M) * H)
                top = int((n - 1) / N * W)
                bottom = int((n / N) * W)
                region = img[left: right, top: bottom]
                region_list.append(region.flatten())

        for k in K:
            gray_levels = (np.sum(region == k) for region in region_list)
            histogram[k] = list(gray_levels)

        return histogram

    def spatialEntropy(self, histogram):
        """
        spatial entropy and distribution function
        :return:
        """
        entropy = dict()  # entropy meature S_k is computed for gray-level x_k
        f_k = dict()  # compute a discrete function f_k
        f_k_norm = dict()  # normalize
        F_cdf = dict()  # cumulative distribution function

        for key, val in histogram.items():
            S_k = 0.0
            val = val / (sum(val) + self._EPS)  # normilize-> very important
            for ele in val:
                if ele != 0:
                    S_k += -(ele * np.log2(ele))  # equation 3
            entropy[key] = S_k
        sum_entropy = sum(entropy.values())

        for key, val in entropy.items():
            f_k[key] = val / ((sum_entropy - val) + self._EPS)  # equation 4
        sum_f_k = sum(f_k.values())

        for key, val in f_k.items():
            f_k_norm[key] = val / (sum_f_k + self._EPS)  # equation 5

        values = list(f_k_norm.values())
        for index, key in enumerate(f_k_norm.keys()):
            F_cdf[key] = sum(values[:(index + 1)])  # equation 6

        return f_k_norm, F_cdf

    def mapping(self, cdf, yd, yu):
        """
        mapping function: using the cumulative distribution function
        :return:
        """
        ymap = dict()
        for key, val in cdf.items():
            ymap[key] = int(np.rint(val * (yu - yd) + yd))  # equation 7
        return ymap

    def pixelMapping(self, img_hsv, mapping):
        '''
        get the enhanced image
        :param img_gray:
        :param map:
        :return:
        '''
        # img = img_hsv[:, :, 2]
        # h = img.shape[0]; w = img.shape[1]
        # V = np.array(img).flatten()
        # V = map(lambda x: mapping[x], V)
        # global_img = np.array(list(V)).reshape(h, w)

        img = img_hsv[:, :, 2]
        lutData = []
        for key, value in enumerate(mapping.items()):
            lutData.append(value[1])
        global_img = np.zeros_like(img)
        table = np.array(lutData).clip(0,255).astype('uint8')
        cv2.LUT(img, table, global_img)

        return global_img

    def dctTransform(self, img):
        '''
        forward 2D-DCT transform
        :param img_hsv: hsv color space
        :param fk: the discrete function
        :return:
        '''
        img = np.float32(img)
        D = cv2.dct(img)  # equation 8, 9

        return D

    def domainCoefWeight(self, dkl_img, fk, gamma=0.25):
        '''
         transform domain coefficient weighting
        :param dkl_img:
        :param fk:
        :return:
        '''
        H = dkl_img.shape[0]
        W = dkl_img.shape[1]

        sum = 0
        for key, value in enumerate(fk.items()):
            if value[1] != 0:
                sum += -value[1] * np.log2(value[1])  # equation 12

        alpha = np.float_power(sum, gamma)

        ww = np.linspace(1, alpha, W).reshape(1, W)  # equation 11
        wh = np.linspace(1, alpha, H).reshape(1, H)
        weight = wh.T * ww  # equation 10
        D_w = np.multiply(dkl_img, weight)

        return D_w

    def inverseDct(self, D_w):
        '''
         inverse 2D-DCT transform
        :param D_w:
        :return:
        '''

        iDct = cv2.idct(D_w)  # equation 8, 9
        Y = np.clip(np.abs(iDct), 0, 255)

        return Y

    def color_restoration(self, S):
        '''
        restore the image color
        :param S:
        :param lambdaa:
        :return:
        '''
        HSV = self.img_hsv
        HSV[:, :, 2] = S
        S_restore = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
        S_restore = cv2.resize(S_restore, (self.width, self.height), cv2.INTER_AREA)

        return np.clip(S_restore, 0, 255).astype('uint8')

    def pltHist(self, img_gray, new_img, str='SECEDCT'):
        '''
        plot Hist of the gray image
        :param img_gray:
        :param new_img:
        :param str:
        :return:
        '''
        fig = plt.figure(figsize=(16, 10))
        ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
        ax1.hist(img_gray.ravel(), 256, [0, 256])
        ax2.hist(new_img.ravel(), 256, [0, 256])
        ax1.set_title("raw image")
        ax2.set_title("the enhanced image with {}".format(str))

    def pltEnhanceImg(self, img_bgr, new_img, str='SECEDCT'):
        '''
        plot the enhanced image
        :param img_gray:
        :param new_img:
        :return:
        '''
        fig1 = plt.figure(figsize=(16, 10))
        ax1, ax2 = fig1.add_subplot(121), fig1.add_subplot(122)
        ax1.imshow(img_bgr[:, :, [2, 1, 0]])
        ax1.axis('off')
        ax2.imshow(new_img[:, :, [2, 1, 0]])
        ax2.axis('off')
        ax1.set_title("raw image")
        ax2.set_title("the enhanced image with {}".format(str))

def secedct(IMG_DIR, IMG_NAME, gamma=0.25):
    '''
    the entire image contrast enhancement process
    :param IMG_DIR:
    :param IMG_NAME:
    :param gamma: control the detail of image, gamma\in [0,1]
    :return: enhanced image and the QRCM value
    '''
    if gamma > 1 or gamma < 0:
        print("The input image exceeds the limitation in the range of [0,1], \n"
              "the program automatically set the gamma as 0.25")
        gamma = 0.25

    # the class of the algorithm named SECEDCT
    secedct = SECEDCT(path.join(IMG_DIR, IMG_NAME))
    # the size of raw image
    img_hsv = secedct.img_hsv

    hist = secedct.spatialHistgram(img_hsv)
    fk, cdf = secedct.spatialEntropy(hist)
    gray_leval_map = secedct.mapping(cdf, yd=0.0, yu=255.0)
    global_img = secedct.pixelMapping(img_hsv, gray_leval_map)
    D = secedct.dctTransform(global_img)
    D_w = secedct.domainCoefWeight(D, fk, gamma)
    invDct_img = secedct.inverseDct(D_w)
    secedct_img = secedct.color_restoration(invDct_img)

    return secedct_img

def qrcm(IMG_DIR, IMG_NAME, secedct_img):
    """
    calculate the QRCM value in the range of [-1,1]
    :param IMG_DIR: the directory of raw image
    :param IMG_NAME: the raw image name
    :param secedct_img: the image enhanced by secedct algorithm
    :return: the qrcm value
    """
    raw_img = cv2.imread(os.path.join(IMG_DIR, IMG_NAME))
    secedct_qrcm = QRCM().qrcm(raw_img, secedct_img)

    return secedct_qrcm

def showIMG(IMG_DIR, IMG_NAME, newImg):
    """
    show the raw image and enhanced image
    :param IMG_DIR: the directory of raw image
    :param IMG_NAME: the raw image name
    :param newImg: the enhanced image
    :return:
    """
    rawImg = cv2.imread(os.path.join(IMG_DIR, IMG_NAME))
    cv2.imshow(IMG_NAME, rawImg)
    cv2.imshow("SECEDCT", newImg)

    print("Please press 'q' and exit !!!")
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()




























