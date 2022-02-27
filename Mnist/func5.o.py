#@ type: compute
#@ parents:
#@   - func1
#@   - func2
#@   - func3
#@   - func4
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
        
    # Get image length
    trans.read(4, 0, 0)
    print("Get image length")
    imglength = struct.unpack_from('@I', trans.buf[0:4])[0]
    print(imglength)
    remoteIndex = 4
    remoteIndex += imglength
    
    # Get label length
    trans.read(4, remoteIndex, 0)
    print("Get label length")
    seclength = struct.unpack_from('@I', trans.buf[0:4])[0]
    remoteIndex += 4
    print(seclength)
    remoteIndex += seclength

    # Fetch boxes data from remote memory server
    trans.read(4, remoteIndex, 0)
    print("Get boxes data length")
    thirdlength = struct.unpack_from('@I', trans.buf[0:4])[0]
    remoteIndex += 4
    print(thirdlength)

    begin = 0
    blockSize = 1000000
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

    # Fetch predictions data from remote memory server
    trans.read(4, remoteIndex, 0)
    print("Get predictions data length")
    predictlength = struct.unpack_from('@I', trans.buf[0:4])[0]
    remoteIndex += 4
    print(predictlength)

    begin = 0
    while begin + blockSize < predictlength:
          trans.read(blockSize, remoteIndex, begin)
          begin += blockSize
          remoteIndex += blockSize
    trans.read(predictlength - begin, remoteIndex, begin)
    remoteIndex += predictlength - begin
    print("Finish read boxes data")
    predictBytes = trans.buf[0:predictlength]
    predictions = pickle.loads(predictBytes)

    # serialize predictions data
    predictBytes = pickle.dumps(predictions)
    length = len(predictBytes)
    struct.pack_into('@I', trans.buf, 0, length)
    trans.write(4, remoteIndex, 0)
    remoteIndex += 4

    filename = "data/1234567890.png"
    output_name = "data/1234567890_ocr.png"

    img = cv2.imread(filename)
    h, w, _ = img.shape
    
    for box, prediction in zip(boxes, predictions):
        img = cv2.rectangle(img, (box.wmin, box.hmin), (box.wmax, box.hmax), (87, 201, 0), 2)
        img = cv2.putText(img, str(prediction), ((box.wmin + box.wmax) // 2, box.hmin), cv2.FONT_HERSHEY_COMPLEX, min(h, w) / 500, (255, 144, 30), 2)
    
    cv2.imwrite(output_name, img)

    print("Finish output image")

    return {}
