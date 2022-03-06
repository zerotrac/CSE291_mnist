#@ type: compute
#@ parents:
#@   - func1
#@ dependents:
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
    


def main(params, action):

    labels = list()

    filename = "data/mnist_labels"

    item_count = TRAIN_SIZE

    trans = action.get_transport('mem1', 'rdma')
    trans.reg(buffer_pool_lib.buffer_size)
    remoteIndex = 0
    count = 0
    beginIndex = getNextIndex(trans, remoteIndex)
    remoteIndex = beginIndex + 4
    

    with open(filename, "rb") as fin:
        assert int.from_bytes(fin.read(4), byteorder="big") == 0x00000801
        N = int.from_bytes(fin.read(4), byteorder="big")
        assert N == item_count
        #assert (N := int.from_bytes(fin.read(4), byteorder="big")) == item_count

        for t in range(N):
            if t % 1000 == 0:
                print(f"read_mnist_label() index = {t}")
            if t % BATCH_SIZE == 0 and t != 0:
               remoteIndex = sendToServer(trans, labels, remoteIndex)
               count += 1
               labels.clear()
                

            label = int.from_bytes(fin.read(1), byteorder="big")
            labels.append(label)
        
        if len(labels) != 0:
            remoteIndex = sendToServer(trans, labels, remoteIndex)
            count += 1
            labels.clear()

        struct.pack_into('@I', trans.buf, 0, count)
        trans.write(4, beginIndex, 0)
        return {}
