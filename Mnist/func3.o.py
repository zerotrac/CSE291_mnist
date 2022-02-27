#@ type: compute
#@ parents:
#@   - func1
#@   - func2
#@ dependents:
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

    filename = "data/1234567890.png"

    img = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    h, w = img.shape
    boxes = list()

    # 通过广度优先搜索，定位所有的连通区域
    for i in range(h):
        for j in range(w):
            if img[i][j] == 0:
                count = 0
                q = deque([(i, j)])
                img[i][j] = 1

                hmin = hmax = i
                wmin = wmax = j

                while q:
                    count += 1

                    x, y = q.popleft()
                    hmin = min(hmin, x)
                    hmax = max(hmax, x)
                    wmin = min(wmin, y)
                    wmax = max(wmax, y)

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < h and 0 <= ny < w and img[nx][ny] == 0:
                                q.append((nx, ny))
                                img[nx][ny] = 1
    
                if count >= OCR_THRESHOLD:
                    box_img = img[hmin:hmax+1, wmin:wmax+1]

                    # 计算 resize 的参数
                    span = SIZE - BORDER * 2
                    hdiff = hmax - hmin + 1
                    wdiff = wmax - wmin + 1
                    maxdiff = max(hdiff, wdiff)

                    hdiff = round(hdiff / maxdiff * span)
                    if hdiff % 2 == 1:
                        hdiff += 1
                    wdiff = round(wdiff / maxdiff * span)
                    if wdiff % 2 == 1:
                        wdiff += 1
                    
                    hborder = (28 - hdiff) // 2
                    wborder = (28 - wdiff) // 2

                    box_img = cv2.resize(box_img, dsize=(wdiff, hdiff), interpolation=cv2.INTER_AREA)

                    # 将 resize 好的 box_img 填入 sub_img
                    sub_img = np.full(shape=(SIZE, SIZE), fill_value=255, dtype=np.uint8)
                    sub_img[hborder:SIZE-hborder, wborder:SIZE-wborder] = box_img

                    sub_img = cv2.adaptiveThreshold(sub_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

                    compressed_img = sub_img.tolist()
                    # mnist 背景为黑色（二值化为 0），数字为白色（二值化为 1）
                    # 而图片一般背景为白色（二值化为 255），数字为黑色（二值化为 0）
                    # 所以要将图片取反
                    for x in range(SIZE):
                        for y in range(SIZE):
                            compressed_img[x][y] = int(compressed_img[x][y] == 0)
                    
                    compressed_img = compression(compressed_img)

                    box = Box(wmin, hmin, wmax, hmax, compressed_img)
                    boxes.append(box.serialize())
    
    print(f"read_OCR_image(): found {len(boxes)} boxes")
        
    trans = action.get_transport('mem1', 'rdma')
    trans.reg(buffer_pool_lib.buffer_size)
 
    # Get image length from memory server
    trans.read(4, 0, 0)
    print("Get image length")
    remoteIndex = struct.unpack_from('@I', trans.buf[0:4])[0]
    print(remoteIndex)
    remoteIndex += 4

    # Get label length from memory server  
    trans.read(4, remoteIndex, 0)
    remoteIndex += 4
    labelLen = struct.unpack_from('@I', trans.buf[0:4])[0]
    print("Get label length")
    print(labelLen)
    remoteIndex += labelLen

    # serialize boxes
    boxesBytes = pickle.dumps(boxes)
    length = len(boxesBytes)
    struct.pack_into('@I', trans.buf, 0, length)
    print(length)
    trans.write(4, remoteIndex, 0)
    remoteIndex += 4
    print("Finish append length")

    # append boxes to local buffer
    trans.buf[0:length] = boxesBytes

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
