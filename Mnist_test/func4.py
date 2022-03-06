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
import pickle
from typing import List
from collections import Counter
import numpy as np

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

OUTPUT = "data/predict_store"

INPUT1 = "data/image_store"

INPUT2 = "data/labels_store"

INPUT3 = "data/boxes_store"

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

def sendToServer(output, data, remoteIndex):
    with open(output, "r+b") as fin:
         dataBytes = pickle.dumps(data)
         length = len(dataBytes)
         lenBytes = struct.pack('@I', length)
         fin.seek(remoteIndex, 0)
         print(fin.tell())
         fin.write(lenBytes)
         print(fin.tell())
         remoteIndex += 4
         fin.seek(remoteIndex, 0)
         fin.write(dataBytes)
         print(fin.tell())
         remoteIndex += length
    return remoteIndex

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
        
    remoteIndexImage = 0
    remoteIndexLabel = 0
    remoteIndexBox = 0

    imageBlockCount = 0
    labelBlockCount = 0
    boxBlockCount = 0

    with open(INPUT1, "r+b") as fin:
         fin.seek(remoteIndexImage, 0)
         countBytes = fin.read(4)
         remoteIndexImage += 4
         imageBlockCount = struct.unpack('@I', countBytes)[0] 
    
    with open(INPUT2, "r+b") as fin:
         fin.seek(remoteIndexLabel, 0)
         countBytes = fin.read(4)
         remoteIndexLabel += 4
         labelBlockCount = struct.unpack('@I', countBytes)[0]

    with open(INPUT3, "r+b") as fin:
         fin.seek(remoteIndexBox, 0)
         countBytes = fin.read(4)
         remoteIndexBox += 4
         boxBlockCount = struct.unpack('@I', countBytes)[0] 

    predictions = list()
    with open(INPUT1, "r+b") as fin1:
         with open(INPUT2, "r+b") as fin2:
              with open(INPUT3, "r+b") as fin3:
                   while boxBlockCount > 0:
                        boxes, remoteIndexBox = getFromServer(INPUT3, remoteIndexBox)
                        for i, boxByte in enumerate(boxes):
                            print(f"knn(): predicting {i + 1} / {len(boxes)} sub-image")
                            box = Box.deserialize(boxByte)
              
                            distances = list()
                            Count = imageBlockCount
                            Index1 = remoteIndexImage
                            Index2 = remoteIndexLabel
                            while Count > 0:
                                  images, Index1 = getFromServer(INPUT1, Index1)
                                  labels, Index2 = getFromServer(INPUT2, Index2)
                                  for img, label in zip(images, labels):
                                      distances.append((compressed_iou(img, box.compressed_img), label))
                                  Count -= 1
                            distances.sort(reverse=True)
                            frequency = Counter(entry[1] for entry in distances[:K])
                            predict = frequency.most_common(1)[0][0]
                            predictions.append(predict)
                        boxBlockCount -= 1

    print("Finish predicting")
    print(predictions)

    count = 0
    remoteIndex = 0
    with open(OUTPUT, "r+b") as fin:
         fin.seek(remoteIndex, 0)
         countBytes = struct.pack('@I', count)
         fin.write(countBytes)
         remoteIndex += 4
    

    predicts = list()
    for predict in predictions:
         predicts.append(predict)
         if len(predicts) == BATCH_SIZE:
              count += 1
              remoteIndex = sendToServer(OUTPUT, predicts, remoteIndex)
              predicts.clear()
    if len(predicts) != 0:
         count += 1
         remoteIndex = sendToServer(OUTPUT, predicts, remoteIndex)
         predicts.clear()

    print(count)
    with open(OUTPUT, "r+b") as fin:
         fin.seek(0, 0)
         fin.write(struct.pack('@I', count))

    return {}

if __name__ == "__main__":
     test()
