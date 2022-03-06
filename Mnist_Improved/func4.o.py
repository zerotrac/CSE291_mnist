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
#@   mem2:
#@     trans: mem2
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

BATCH_SIZE = 1000

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

def sendToServer(trans, data, remoteIndex):
    dataBytes = pickle.dumps(data)
    length = len(dataBytes)
    
    struct.pack_into('@I', trans.buf, 0, length)
    print(length)
    print("Finish append length")
    trans.write(4, remoteIndex, 0)
    remoteIndex += 4
    
    trans.buf[0:length] = dataBytes
    print("Finish append data")

    begin = 0
    blockSize = 1000000
    while begin + blockSize < length:
          trans.write(blockSize, remoteIndex, begin)
          begin += blockSize
          remoteIndex += blockSize
    trans.write(length - begin, remoteIndex, begin)
    remoteIndex += (length - begin)
    print("Finish write data")
    return remoteIndex

def getNextIndex(trans, remoteIndex):
    trans.read(4, remoteIndex, 0)
    count = struct.unpack_from('@I', trans.buf[0:4])[0]
    remoteIndex += 4
    while count > 0:
         trans.read(4, remoteIndex, 0)
         remoteIndex += struct.unpack_from('@I', trans.buf[0:4])[0]
         remoteIndex += 4
         count -= 1
         print(remoteIndex)
    return remoteIndex

def getFromServer(trans, remoteIndex):
    trans.read(4, remoteIndex, 0)
    length = struct.unpack_from('@I', trans.buf[0:4])[0]
    remoteIndex += 4
    blockSize = 1000000
    begin = 0
    while begin + blockSize < length:
          trans.read(blockSize, remoteIndex, begin)
          begin += blockSize
          remoteIndex += blockSize
    trans.read(length - begin, remoteIndex, begin)
    remoteIndex += (length - begin)
    dataBytes = trans.buf[0:length]
    data = pickle.loads(dataBytes)
    return data, remoteIndex


def main(params, action):

    trans1 = action.get_transport('mem1', 'rdma')
    trans1.reg(buffer_pool_lib.buffer_size)

    trans2 = action.get_transport('mem2', 'rdma')
    trans2.reg(buffer_pool_lib.buffer_size)
        
    remoteIndexImage = 0
    remoteIndexLabel = getNextIndex(trans1, 0)
    remoteIndexBox = 0

    trans1.read(4, remoteIndexImage, 0)
    remoteIndexImage += 4
    imageBlockCount = struct.unpack_from('@I', trans1.buf[0:4])[0]

    trans1.read(4, remoteIndexLabel, 0)
    remoteIndexLabel += 4
    labelBlockCount = struct.unpack_from('@I', trans1.buf[0:4])[0]

    trans2.read(4, remoteIndexBox, 0)
    remoteIndexBox += 4
    boxBlockCount = struct.unpack_from('@I', trans2.buf[0:4])[0]    

    predictions = list()
    while boxBlockCount > 0:
          boxes, remoteIndexBox = getFromServer(trans2, remoteIndexBox)
          for i, boxByte in enumerate(boxes):
              print(f"knn(): predicting {i + 1} / {len(boxes)} sub-image")
              box = Box.deserialize(boxByte)
              
              distances = list()
              Count = imageBlockCount
              Index1 = remoteIndexImage
              Index2 = remoteIndexLabel
              while Count > 0:
                    images, Index1 = getFromServer(trans1, Index1)
                    labels, Index2 = getFromServer(trans1, Index2)
                    for img, label in zip(images, labels):
                        distances.append((compressed_iou(img, box.compressed_img), label))
                    Count -= 1
              distances.sort(reverse=True)
              frequency = Counter(entry[1] for entry in distances[:K])
              predict = frequency.most_common(1)[0][0]
              predictions.append(predict)
          boxBlockCount -= 1
    #for i, box in enumerate(boxes):
    #    print(f"knn(): predicting {i + 1} / {len(boxes)} sub-image")
    #
    #    distances = list()
    #    for img, label in zip(images, labels):
    #        distances.append((compressed_iou(img, box.compressed_img), label))
    #    
    #    distances.sort(reverse=True)
    #    frequency = Counter(entry[1] for entry in distances[:K])
    #    predict = frequency.most_common(1)[0][0]
    #    predictions.append(predict)

    print("Finish predicting")
    print(predictions)

    count = 0

    predicts = list()
    predictBeginIndex = remoteIndexBox
    remoteIndexBox += 4
    for predict in predictions:
         predicts.append(predict)
         if len(predicts) == BATCH_SIZE:
              count += 1
              remoteIndexBox = sendToServer(trans2, predicts, remoteIndexBox)
              predicts.clear()
    if len(predicts) != 0:
         count += 1
         remoteIndexBox = sendToServer(trans2, predicts, remoteIndexBox)
         predicts.clear()

    print(count)
    struct.pack_into('@I', trans2.buf, 0, count)
    trans2.write(4, predictBeginIndex, 0)

    return {}
