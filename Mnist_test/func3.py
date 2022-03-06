#@ type: compute
#@ parents:
#@   - func1
#@   - func2
#@ dependents:
#@   - func4
#@ corunning:
#@   mem2:
#@     trans: mem2
#@     type: rdma

import struct
import pickle
from collections import deque, Counter
from typing import List
import cv2
import numpy as np
import pickle

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

OUTPUT = "data/boxes_store"

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

def test():

    filename = "data/1234567890.png"

    img = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    h, w = img.shape
    boxes = list()

    boxNum = 0
    batchCount = 0
    remoteIndex = 0
    with open(OUTPUT, "r+b") as fin:
         fin.seek(remoteIndex, 0)
         countBytes = struct.pack('@I', batchCount)
         fin.write(countBytes)
         remoteIndex += 4

    # 通过广度优先搜索，定位所有的连通区域
    for i in range(h):
        for j in range(w):
            if boxNum % BATCH_SIZE == 0 and boxNum != 0:
                remoteIndex = sendToServer(OUTPUT, boxes, remoteIndex)
                batchCount += 1
                boxes.clear()
                
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
                    boxNum += 1
    
    if len(boxes) != 0:
        remoteIndex = sendToServer(OUTPUT, boxes, remoteIndex)
        batchCount += 1
        boxes.clear()

    with open(OUTPUT, "r+b") as fin:
         fin.seek(0, 0)
         fin.write(struct.pack('@I', batchCount))
    # print(f"read_OCR_image(): found {len(boxes)} boxes")
        

    print("Finish write data")
    return {}

if __name__ == "__main__":
     test()
