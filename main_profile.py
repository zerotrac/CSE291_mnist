from collections import deque, Counter
from glob import glob
from typing import List
import cv2
import numpy as np
import pickle
from pympler import asizeof
import time

import const
from distance_function import compressed_iou
from image_operation import binarization, compression


class Box:
    def __init__(self, wmin: int, hmin: int, wmax: int, hmax: int, compressed_img: List[int]):
        self.wmin = wmin
        self.hmin = hmin
        self.wmax = wmax
        self.hmax = hmax
        self.compressed_img = compressed_img


train_images = list()
train_labels = list()
boxes = list()
img = list()
predictions = list()

p_train_images = None
p_train_labels = None

last_counter = 0.0
fout = open("memory.dat", "w")

def profile() -> None:
    global train_images, train_labels, boxes, img, predictions
    global last_counter

    t = time.perf_counter()
    if t - last_counter >= 0.1:
        result = asizeof.asizesof(train_images, train_labels, boxes, img, predictions)
        print(t - last_counter, result)
        print(*result, file=fout)
        last_counter = time.perf_counter()


def read_mnist_image(filename: str, item_count: int) -> None:
    global train_images, p_train_images

    with open(filename, "rb") as fin:
        assert int.from_bytes(fin.read(4), byteorder="big") == 0x00000803
        assert (N := int.from_bytes(fin.read(4), byteorder="big")) == item_count
        assert int.from_bytes(fin.read(4), byteorder="big") == const.SIZE
        assert int.from_bytes(fin.read(4), byteorder="big") == const.SIZE

        for t in range(N):
            
            profile()

            if t % 1000 == 0:
                print(f"read_mnist_image() index = {t}")

            img = [[0] * const.SIZE for _ in range(const.SIZE)]

            # 依次读入图片的每一个 byte
            for i in range(const.SIZE):
                for j in range(const.SIZE):
                    img[i][j] = int.from_bytes(fin.read(1), byteorder="big")
            
            # 二值化 + 压缩
            img = binarization(img)
            img = compression(img)

            train_images.append(img)
    
    p_train_images = pickle.dumps(train_images)
    train_images = list()


def read_mnist_label(filename: str, item_count: int) -> None:
    global train_labels, p_train_labels

    with open(filename, "rb") as fin:
        assert int.from_bytes(fin.read(4), byteorder="big") == 0x00000801
        assert (N := int.from_bytes(fin.read(4), byteorder="big")) == item_count

        for t in range(N):
        
            profile()

            if t % 1000 == 0:
                print(f"read_mnist_label() index = {t}")

            label = int.from_bytes(fin.read(1), byteorder="big")
            train_labels.append(label)
    
    p_train_labels = pickle.dumps(train_labels)
    train_labels = list()


def read_OCR_image(filename: str) -> None:
    global boxes, img

    profile()

    # 直接读入灰度图
    img = cv2.imread(filename, flags=cv2.IMREAD_GRAYSCALE)

    profile()

    img = cv2.medianBlur(img, 5)

    profile()
    
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    profile()

    h, w = img.shape

    # 通过广度优先搜索，定位所有的连通区域
    for i in range(h):
        for j in range(w):
            
            profile()

            if img[i][j] == 0:
                count = 0
                q = deque([(i, j)])
                img[i][j] = 1

                hmin = hmax = i
                wmin = wmax = j

                while q:

                    profile()

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
    
                profile()

                if count >= const.OCR_THRESHOLD:
                    box_img = img[hmin:hmax+1, wmin:wmax+1]

                    # 计算 resize 的参数
                    span = const.SIZE - const.BORDER * 2
                    hdiff = hmax - hmin + 1
                    wdiff = wmax - wmin + 1
                    maxdiff = max(hdiff, wdiff)

                    hdiff = max(round(hdiff / maxdiff * span), 2)
                    if hdiff % 2 == 1:
                        hdiff += 1
                    wdiff = max(round(wdiff / maxdiff * span), 2)
                    if wdiff % 2 == 1:
                        wdiff += 1
                    
                    hborder = (28 - hdiff) // 2
                    wborder = (28 - wdiff) // 2

                    profile()

                    box_img = cv2.resize(box_img, dsize=(wdiff, hdiff), interpolation=cv2.INTER_AREA)

                    profile()

                    # 将 resize 好的 box_img 填入 sub_img
                    sub_img = np.full(shape=(const.SIZE, const.SIZE), fill_value=255, dtype=np.uint8)
                    sub_img[hborder:const.SIZE-hborder, wborder:const.SIZE-wborder] = box_img

                    profile()

                    sub_img = cv2.adaptiveThreshold(sub_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
                    
                    profile()

                    compressed_img = sub_img.tolist()
                    # mnist 背景为黑色（二值化为 0），数字为白色（二值化为 1）
                    # 而图片一般背景为白色（二值化为 255），数字为黑色（二值化为 0）
                    # 所以要将图片取反
                    for x in range(const.SIZE):
                        for y in range(const.SIZE):
                            compressed_img[x][y] = int(compressed_img[x][y] == 0)
                    
                    profile()
                    
                    compressed_img = compression(compressed_img)

                    profile()

                    box = Box(wmin, hmin, wmax, hmax, compressed_img)
                    boxes.append(box)
    
    print(f"read_OCR_image(): found {len(boxes)} boxes")

    img = list()


def knn() -> List[int]:
    global train_images, train_labels, predictions, p_train_images, p_train_labels

    train_images = pickle.loads(p_train_images)
    train_labels = pickle.loads(p_train_labels)

    for i, box in enumerate(boxes):

        profile()

        print(f"knn(): predicting {i + 1} / {len(boxes)} sub-image")

        distances = list()
        for img, label in zip(train_images, train_labels):

            profile()

            distances.append((compressed_iou(img, box.compressed_img), label))
        
        distances.sort(reverse=True)

        profile()

        frequency = Counter(entry[1] for entry in distances[:const.K])
        predict = frequency.most_common(1)[0][0]
        predictions.append(predict)
    
    train_images = list()
    train_labels = list()


def collect_result(filename: str, output_name: str):
    global boxes, img, predictions

    profile()

    img = cv2.imread(filename)
    h, w, _ = img.shape

    profile()
    
    for box, prediction in zip(boxes, predictions):
        
        profile()
        
        img = cv2.rectangle(img, (box.wmin, box.hmin), (box.wmax, box.hmax), (87, 201, 0), 2)

        profile()
        
        img = cv2.putText(img, str(prediction), ((box.wmin + box.wmax) // 2, box.hmin), cv2.FONT_HERSHEY_COMPLEX, min(h, w) / 500, (255, 144, 30), 2)
    
    cv2.imwrite(output_name, img)

    profile()


if __name__ == "__main__":
    # 第一步：读 mnist 训练集的图片，将图片二值化、压缩后进行存储，节省空间
    read_mnist_image("data/mnist_images", const.TRAIN_SIZE)

    # 第二步：读 mnist 训练集的标注
    read_mnist_label("data/mnist_labels", const.TRAIN_SIZE)

    # 第三步：读待 OCR 的图片，将图片通过一系列 opencv 函数进行处理，以找出图片中的每一个数字，将它们 resize 后进行存储，同时存储它们在原图片中的位置
    read_OCR_image("data/1234567890.png")
    
    # 第四步：对于每一个数字，使用 KNN 来判断它是哪一个数字
    knn()

    # 第五步：重新打开待 OCR 的图片，框出所有数字（根据第三步存储的数据）并标出（根据第四步的结果）
    collect_result("data/1234567890.png", "data/1234567890_ocr.png")

    fout.close()