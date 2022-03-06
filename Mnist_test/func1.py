#@ type: compute
#@ dependents:
#@   - func2
#@   - func3
#@ corunning:
#@   mem1:
#@     trans: mem1
#@     type: rdma


import struct
import pickle
from typing import List

TRAIN_SIZE = 60000
SIZE = 28

BATCH_SIZE = 1000

OUTPUT = "data/image_store"

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

    images = list()
    filename = "data/mnist_images"
    item_count = TRAIN_SIZE

    remoteIndex = 0
    count = 0
    with open(OUTPUT, "r+b") as fin:
         fin.seek(remoteIndex, 0)
         countBytes = struct.pack('@I', count)
         fin.write(countBytes)
         remoteIndex += 4

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
                remoteIndex = sendToServer(OUTPUT, images, remoteIndex)
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
             remoteIndex = sendToServer(OUTPUT, images, remoteIndex)
             count += 1
             images.clear()

    with open(OUTPUT, "r+b") as fin:
         fin.seek(0, 0)
         fin.write(struct.pack('@I', count))
    return {}


if __name__ == "__main__":
     test()
