#@ type: compute
#@ parents:
#@   - func1
#@   - func2
#@   - func3
#@   - func4
#@ corunning:
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

    trans = action.get_transport('mem2', 'rdma')
    trans.reg(buffer_pool_lib.buffer_size)

    remoteIndexBox = 0
    remoteIndexPre = getNextIndex(trans, remoteIndexBox)
    remoteIndexPre += 4
    trans.read(4, remoteIndexBox, 0)
    remoteIndexBox += 4
    boxBlockCount = struct.unpack_from('@I', trans.buf[0:4])[0]

    filename = "data/1234567890.png"
    output_name = "data/1234567890_ocr.png"

    img = cv2.imread(filename)
    h, w, _ = img.shape

    while boxBlockCount > 0:
         boxes, remoteIndexBox = getFromServer(trans, remoteIndexBox)
         predictions, remoteIndexPre = getFromServer(trans, remoteIndexPre)
         for boxByte, prediction in zip(boxes, predictions):
              box = Box.deserialize(boxByte)
              img = cv2.rectangle(img, (box.wmin, box.hmin), (box.wmax, box.hmax), (87, 201, 0), 2)
              img = cv2.putText(img, str(prediction), ((box.wmin + box.wmax) // 2, box.hmin), cv2.FONT_HERSHEY_COMPLEX, min(h, w) / 500, (255, 144, 30), 2)
         boxBlockCount -= 1
    
    cv2.imwrite(output_name, img)

    print("Finish output image")

    return {}
