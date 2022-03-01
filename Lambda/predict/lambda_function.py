import json
import numpy as np
import cv2
import boto3
import pickle
import const

from collections import deque, Counter
from distance_function import compressed_iou
from typing import List

s3_client = boto3.client('s3')

class Box:
    def __init__(self, wmin: int, hmin: int, wmax: int, hmax: int, compressed_img: List[int]):
        self.wmin = wmin
        self.hmin = hmin
        self.wmax = wmax
        self.hmax = hmax
        self.compressed_img = compressed_img

def lambda_handler(event, context):
    mnist_bucket = "processed-mnist-training"
    mnist_image_key = "mnist_images"
    mnist_image_download_path = '/tmp/mnist_images'
    s3_client.download_file(mnist_bucket, mnist_image_key, mnist_image_download_path)
    print("mnist image downloaded from {}".format(mnist_image_download_path))
    images = get_information(mnist_image_download_path)

    mnist_label_key = "mnist_labels"
    mnist_label_download_path = '/tmp/mnist_labels'
    s3_client.download_file(mnist_bucket, mnist_label_key, mnist_label_download_path)
    print("mnist label downloaded from {}".format(mnist_label_download_path))
    labels = get_information(mnist_label_download_path)

    for record in event["Records"]:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        download_path = '/tmp/ocr-{}'.format(key)

        upload_path = '/tmp/predict-{}'.format(key)
        s3_client.download_file(bucket, key, download_path)
        print("ocr result input {} downloaded from {}".format(key ,bucket))

        predictInput(images, labels, download_path, upload_path)

        s3_client.upload_file(upload_path, const.S3predict, key)
        print("new result uploaded to {}".format(const.S3predict))


def predictInput(images, labels, download_path, upload_path):
    with open(download_path, 'rb') as f:
        boxes = pickle.load(f)
        results = knn(images, labels, boxes)

    with open(upload_path, 'wb') as f:
        pickle.dump(results, f)

def knn(images: List[List[int]], labels: List[int], boxes: List[Box]) -> List[int]:
    predictions = list()

    for i, box in enumerate(boxes):
        print(f"knn(): predicting {i + 1} / {len(boxes)} sub-image")

        distances = list()
        for img, label in zip(images, labels):
            distances.append((compressed_iou(img, box.compressed_img), label))
        
        distances.sort(reverse=True)
        frequency = Counter(entry[1] for entry in distances[:const.K])
        predict = frequency.most_common(1)[0][0]
        predictions.append(predict)
        print(f"knn(): predict {i + 1} / {len(boxes)} sub-image, result {predict}")

    return predictions

def get_information(download_path):
    with open(download_path, "rb") as fin:
        infos = pickle.load(fin)
    return infos