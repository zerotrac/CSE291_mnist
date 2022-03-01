import json
import numpy as np
import cv2
import boto3
import pickle

import const
from typing import List

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    mnist_image_download_path = '/tmp/processed-mnist_images'
    s3_client.download_file(bucket, key, mnist_image_download_path)
    print("training set {} downloaded from {}".format(key ,bucket))
    images = get_information(mnist_image_download_path)

    mnist_label_download_path = '/tmp/processed-mnist_labels'
    s3_client.download_file(bucket, key, mnist_label_download_path)
    print("training set {} downloaded from {}".format(key ,bucket))
    labels = get_information(mnist_label_download_path)

    for record in event["Records"]:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        download_path = '/tmp/ocr-{}'.format(key)

        upload_path = '/tmp/predict-{}'.format(key)
        s3_client.download_file(bucket, key, download_path)
        print("ocr result input {} downloaded from {}".format(key ,bucket))

        predictInput(images, labels, download_path, upload_path)

        s3_client.upload_file(upload_path, const.S3output, key)
        print("new picture uploaded to {}".format(const.S3output))


def pridictInput(images, labels, download_path, upload_path):
    with open(download_path, 'rb') as f:
        boxes = pickle.load(f)
        results = knn(images, labels, boxes)

    with open(upload_path, 'wb') as f:
        pickle.dump(results)
        
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

    return predictions

def get_information(download_path):
    with open(download_path, "rb") as fin:
        infos = pickle.load()