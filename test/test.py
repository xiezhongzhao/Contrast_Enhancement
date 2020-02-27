#!/usr/bin/env python
# encoding: utf-8

# @Author: Xie Zhongzhao
# @Date  : 2019-12-02 10:12:00

import sys
import os
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


parser = argparse.ArgumentParser()
parser.description='please input the image !!!'
parser.add_argument("-name", "--inputA", help="name of input image", dest="argA", type=str, default="0")
parser.add_argument("-gamma", "--inputB", help="control the local contrast", dest="argB", type=float, default="0.25")
args = parser.parse_args()

from ContrastEnhancement.SpatialEntropy import secedct, qrcm, showIMG

if __name__ == '__main__':

    IMG_DIR = os.path.join(rootPath, 'data')
    IMG_NAME = args.argA
    Gamma = args.argB

    secedct_img = secedct(IMG_DIR, IMG_NAME, gamma=Gamma)
    showIMG(IMG_DIR, IMG_NAME, secedct_img)

    secedct_qrcm = qrcm(IMG_DIR, IMG_NAME, secedct_img)
    print("secedct_qrcm value: ", secedct_qrcm)











