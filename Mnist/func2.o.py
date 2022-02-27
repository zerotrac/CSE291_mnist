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

def main(params, action):

    labels = list()

    filename = "data/mnist_labels"

    item_count = TRAIN_SIZE

    with open(filename, "rb") as fin:
        assert int.from_bytes(fin.read(4), byteorder="big") == 0x00000801
        N = int.from_bytes(fin.read(4), byteorder="big")
        assert N == item_count
        #assert (N := int.from_bytes(fin.read(4), byteorder="big")) == item_count

        for t in range(N):
            if t % 1000 == 0:
                print(f"read_mnist_label() index = {t}")

            label = int.from_bytes(fin.read(1), byteorder="big")
            labels.append(label)
        
        trans = action.get_transport('mem1', 'rdma')
        trans.reg(buffer_pool_lib.buffer_size)
 
        # Get meta length of images from memory server
        # We should write our labels data after the images data, and they should not overlap with each other
        trans.read(4, 0, 0)
        print("Get length")
        remoteIndex = struct.unpack_from('@I', trans.buf[0:4])[0]
        remoteIndex += 4

        # serialize labels data
        labelBytes = pickle.dumps(labels)

        length = len(labelBytes)
        struct.pack_into('@I', trans.buf, 0, length)
        print(length)
        print("Finish append length")
        trans.write(4, remoteIndex, 0)
        remoteIndex += 4

        # append labels data to local buffer
        trans.buf[0:length] = labelBytes

        # send labels data to remote memory server
        begin = 0
        blockSize = 1000000
        while begin + blockSize < length:
              trans.write(blockSize, remoteIndex, begin)
              begin += blockSize
              remoteIndex += blockSize
        trans.write(length - begin, remoteIndex, begin)
        
        print("Finish write data")
        return {}
