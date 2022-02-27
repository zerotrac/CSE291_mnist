#@ type: compute
#@ parents:
#@   - func1
#@   - func2
#@   - func3
#@ dependents:
#@   - func5
#@ corunning:
#@   mem1:
#@     trans: mem1
#@     type: rdma


import struct
import threading
import time
import pickle
import sys
import copy
import codecs
import copyreg
import collections
from base64 import b64encode
from collections import deque, Counter
from typing import List
import cv2
import numpy as np
import pickle

import disaggrt.buffer_pool_lib as buffer_pool_lib
from disaggrt.rdma_array import remote_array

# mnist 数据集图片的长宽
SIZE = 28

# mnist 训练集大小
TRAIN_SIZE = 60000

# mnist 测试集大小（用不到）
TEST_SIZE = 10000

# 发现连通的至少多少个颜色块才会被认为是数字
OCR_THRESHOLD = 100

# 在对图片进行 resize 成 SIZE * SIZE 大小的时候，上下左右保留的空白边界大小
BORDER = 4

# KNN 中的超参数
K = 10

def count_one(x: int) -> int:
    """
    计算 x 二进制表示中 1 的个数
    """
    return bin(x).count("1")

def compressed_iou(x: List[int], y: List[int]) -> float:
    """
    用于 KNN 的相似度函数
    x 和 y 是两个压缩好的一维列表，相似度就是交并比
    """
    numerator = denominator = 0
    for xk, yk in zip(x, y):
        numerator += count_one(xk & yk)
        denominator += count_one(xk | yk)
    return numerator / denominator

class Box:
    def __init__(self, wmin: int, hmin: int, wmax: int, hmax: int, compressed_img: List[int]):
        self.wmin = wmin
        self.hmin = hmin
        self.wmax = wmax
        self.hmax = hmax
        self.compressed_img = compressed_img
    def serialize(self):
        return [self.wmin, self.hmin, self.wmax, self.hmax] + self.compressed_img

    @staticmethod
    def deserialize(obj):
        return Box(obj[0], obj[1], obj[2], obj[3], obj[4:])


def main(params, action):

    trans = action.get_transport('mem1', 'rdma')
    trans.reg(buffer_pool_lib.buffer_size)
        
    # Fetch image data from remote memory server
    trans.read(4, 0, 0)
    print("Get image length")
    length = struct.unpack_from('@I', trans.buf[0:4])[0]
    print(length)
    remoteIndex = 4
    begin = 0
    blockSize = 1000000
    while begin + blockSize < length:
          trans.read(blockSize, remoteIndex, begin)
          begin += blockSize
          remoteIndex += blockSize
    trans.read(length - begin, remoteIndex, begin)
    remoteIndex += (length - begin)
    print("Finish read image data")
    imageBytes = trans.buf[0:length]
    images = pickle.loads(imageBytes)
    
    # Fetch label data from remote memory server
    trans.read(4, remoteIndex, 0)
    print("Get label length")
    seclength = struct.unpack_from('@I', trans.buf[0:4])[0]
    remoteIndex += 4
    print(seclength)
    begin = 0
    while begin + blockSize < seclength:
          trans.read(blockSize, remoteIndex, begin)
          begin += blockSize
          remoteIndex += blockSize
    trans.read(seclength - begin, remoteIndex, begin)
    remoteIndex += seclength - begin
    print("Finish read label data")
    labelBytes = trans.buf[0:seclength]
    labels = pickle.loads(labelBytes)

    # Fetch boxes data from remote memory server
    trans.read(4, remoteIndex, 0)
    print("Get boxes data length")
    thirdlength = struct.unpack_from('@I', trans.buf[0:4])[0]
    remoteIndex += 4
    print(thirdlength)
    begin = 0
    while begin + blockSize < thirdlength:
          trans.read(blockSize, remoteIndex, begin)
          begin += blockSize
          remoteIndex += blockSize
    trans.read(thirdlength - begin, remoteIndex, begin)
    remoteIndex += thirdlength - begin
    print("Finish read boxes data")
    boxesBytes = trans.buf[0:thirdlength]
    serializedBoxes = pickle.loads(boxesBytes)
    boxes = list()
    for box in serializedBoxes:
        boxes.append(Box.deserialize(box))

    print("Finish loading all the objects")

    predictions = list()
    for i, box in enumerate(boxes):
        print(f"knn(): predicting {i + 1} / {len(boxes)} sub-image")

        distances = list()
        for img, label in zip(images, labels):
            distances.append((compressed_iou(img, box.compressed_img), label))
        
        distances.sort(reverse=True)
        frequency = Counter(entry[1] for entry in distances[:K])
        predict = frequency.most_common(1)[0][0]
        predictions.append(predict)

    print("Finish predicting")
    print(predictions)

    # serialize predictions data
    predictBytes = pickle.dumps(predictions)
    length = len(predictBytes)
    struct.pack_into('@I', trans.buf, 0, length)
    trans.write(4, remoteIndex, 0)
    remoteIndex += 4

    # append predictions data to local buffer
    trans.buf[0:length] = predictBytes

    # send boxes to remote memory server
    begin = 0
    blockSize = 1000000
    while begin + blockSize < length:
          trans.write(blockSize, remoteIndex, begin)
          begin += blockSize
          remoteIndex += blockSize
    trans.write(length - begin, remoteIndex, begin)
    print("Finish write data")

    return {}
