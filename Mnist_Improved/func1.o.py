#@ type: compute
#@ dependents:
#@   - func2
#@   - func3
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

TRAIN_SIZE = 60000
SIZE = 28

BATCH_SIZE = 1000

def binarization(img: List[List[int]]) -> List[List[int]]:
    """
    对于给定的一张 const.SIZE * const.SIZE 大小的灰度图片
    将其二值化：大于等于平均灰度的置为 1，其它置为 0
    """
    average = sum(sum(row) for row in img) / (SIZE * SIZE)
    for i in range(SIZE):
        for j in range(SIZE):
            img[i][j] = (1 if img[i][j] >= average else 0)
    
    return img

def compression(img: List[List[int]], group_size = 16) -> List[int]:
    """
    对于给定的一张 const.SIZE * const.SIZE 大小的二值化图片
    将其按照行优先的顺序，每 group_size 个 bit 压缩成一个数，得到一个一维列表，便于存储和相似度计算
    """
    span = sum(img, [])
    compressed_img = list()
    for i in range(0, SIZE * SIZE, group_size):
        segment = "".join(str(d) for d in span[i:i+group_size])
        compressed_img.append(int(segment, 2))
        
    return compressed_img

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

def main(params, action):

    images = list()
    filename = "data/mnist_images"
    item_count = TRAIN_SIZE
    action.profile(1)

    trans = action.get_transport('mem1', 'rdma')
    trans.reg(buffer_pool_lib.buffer_size)
    remoteIndex = 4
    count = 0

    with open(filename, "rb") as fin:
        assert int.from_bytes(fin.read(4), byteorder="big") == 0x00000803
        N = int.from_bytes(fin.read(4), byteorder="big")
        assert N == item_count
        assert int.from_bytes(fin.read(4), byteorder="big") == SIZE
        assert int.from_bytes(fin.read(4), byteorder="big") == SIZE

        for t in range(N):
            if t % 1000 == 0:
                print(f"read_mnist_image() index = {t}")
            if t % BATCH_SIZE == 0 and t != 0:
                remoteIndex = sendToServer(trans, images, remoteIndex)
                count += 1
                images.clear()

            img = [[0] * SIZE for _ in range(SIZE)]

            # 依次读入图片的每一个 byte
            for i in range(SIZE):
                for j in range(SIZE):
                    img[i][j] = int.from_bytes(fin.read(1), byteorder="big")
            
            # 二值化 + 压缩
            img = binarization(img)
            img = compression(img)

            images.append(img)

        
        if len(images) != 0:
             remoteIndex = sendToServer(trans, images, remoteIndex)
             count += 1
             images.clear()

        struct.pack_into('@I', trans.buf, 0, count)
        trans.write(4, 0, 0)
        return {}
