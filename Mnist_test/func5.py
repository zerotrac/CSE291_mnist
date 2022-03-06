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
import pickle
from typing import List
import cv2
import numpy as np


INPUT1 = "data/predict_store"
INPUT2 = "data/boxes_store"

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

def getFromServer(output, remoteIndex):
    with open(output, "r+b") as fin:
         fin.seek(remoteIndex, 0)
         lengthBytes = fin.read(4)
         remoteIndex += 4
         length = struct.unpack('@I', lengthBytes)[0]
         dataBytes = fin.read(length)
         data = pickle.loads(dataBytes)
         remoteIndex += length
         return data, remoteIndex

def test():

    remoteIndexBox = 0
    remoteIndexPre = 0
    
    boxBlockCount = 0
    preBlockCount = 0

    with open(INPUT1, "r+b") as fin:
         fin.seek(remoteIndexPre, 0)
         countBytes = fin.read(4)
         remoteIndexPre += 4
         preBlockCount = struct.unpack('@I', countBytes)[0] 
    
    with open(INPUT2, "r+b") as fin:
         fin.seek(remoteIndexBox, 0)
         countBytes = fin.read(4)
         remoteIndexBox += 4
         boxBlockCount = struct.unpack('@I', countBytes)[0]
    
    filename = "data/1234567890.png"
    output_name = "data/1234567890_ocr.png"

    img = cv2.imread(filename)
    h, w, _ = img.shape

    while boxBlockCount > 0:
         boxes, remoteIndexBox = getFromServer(INPUT2, remoteIndexBox)
         predictions, remoteIndexPre = getFromServer(INPUT1, remoteIndexPre)
         for boxByte, prediction in zip(boxes, predictions):
              box = Box.deserialize(boxByte)
              img = cv2.rectangle(img, (box.wmin, box.hmin), (box.wmax, box.hmax), (87, 201, 0), 2)
              img = cv2.putText(img, str(prediction), ((box.wmin + box.wmax) // 2, box.hmin), cv2.FONT_HERSHEY_COMPLEX, min(h, w) / 500, (255, 144, 30), 2)
         boxBlockCount -= 1
    
    cv2.imwrite(output_name, img)

    print("Finish output image")

    return {}

if __name__ == "__main__":
     test()
