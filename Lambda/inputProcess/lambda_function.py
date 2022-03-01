import json
import numpy as np
import cv2
import boto3
import pickle
from typing import List
from collections import deque

import const
from image_operation import compression

s3_client = boto3.client('s3')

class Box:
    def __init__(self, wmin: int, hmin: int, wmax: int, hmax: int, compressed_img: List[int]):
        self.wmin = wmin
        self.hmin = hmin
        self.wmax = wmax
        self.hmax = hmax
        self.compressed_img = compressed_img

def read_OCR_image(filename: str) -> List[Box]:
    # 直接读入灰度图
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
    
                if count >= const.OCR_THRESHOLD:
                    box_img = img[hmin:hmax+1, wmin:wmax+1]

                    # 计算 resize 的参数
                    span = const.SIZE - const.BORDER * 2
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
                    sub_img = np.full(shape=(const.SIZE, const.SIZE), fill_value=255, dtype=np.uint8)
                    sub_img[hborder:const.SIZE-hborder, wborder:const.SIZE-wborder] = box_img

                    sub_img = cv2.adaptiveThreshold(sub_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

                    compressed_img = sub_img.tolist()
                    # mnist 背景为黑色（二值化为 0），数字为白色（二值化为 1）
                    # 而图片一般背景为白色（二值化为 255），数字为黑色（二值化为 0）
                    # 所以要将图片取反
                    for x in range(const.SIZE):
                        for y in range(const.SIZE):
                            compressed_img[x][y] = int(compressed_img[x][y] == 0)
                    
                    compressed_img = compression(compressed_img)

                    box = Box(wmin, hmin, wmax, hmax, compressed_img)
                    boxes.append(box)
    
    print(f"read_OCR_image(): found {len(boxes)} boxes")
    return boxes


# image to boxes
def lambda_handler(event, context):
    for record in event["Records"]:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        download_path = '/tmp/{}'.format(key)
        upload_path = '/tmp/processed-{}'.format(key)
        s3_client.download_file(bucket, key, download_path)
        print("input image {} downloaded from {}".format(key ,bucket))
        
        boxes = read_OCR_image(download_path)
        with open(upload_path, 'wb') as fp:
            pickle.dump(boxes, fp)

        
        s3_client.upload_file(upload_path, const.S3boxes, key)
        print("new input boxes uploaded to {}".format(const.S3boxes))
        
    return {
        'statusCode': 200,
        'body': json.dumps('done!')
    }

