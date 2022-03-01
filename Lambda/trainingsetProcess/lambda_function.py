import json
import numpy as np
import cv2
import boto3
import pickle

import const
from image_operation import binarization, compression

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    for record in event["Records"]:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        download_path = '/tmp/{}'.format(key)
        upload_path = '/tmp/processed-{}'.format(key)
        s3_client.download_file(bucket, key, download_path)
        print("training set {} downloaded from {}".format(key ,bucket))
        
        # images or labels
        names = key.split('_')
        if names[-1] != 'labels':
            processImages(download_path, upload_path)
        else:
            processLabels(download_path, upload_path)

        
        s3_client.upload_file(upload_path, const.S3train, key)
        print("new training set uploaded to {}".format(const.S3name))
        
    return {
        'statusCode': 200,
        'body': json.dumps('done!')
    }

def read_mnist_image(filename: str, item_count: int) -> List[List[int]]:
    images = list()

    with open(filename, "rb") as fin:
        assert int.from_bytes(fin.read(4), byteorder="big") == 0x00000803
        assert (N := int.from_bytes(fin.read(4), byteorder="big")) == item_count
        assert int.from_bytes(fin.read(4), byteorder="big") == const.SIZE
        assert int.from_bytes(fin.read(4), byteorder="big") == const.SIZE

        for t in range(N):
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

            images.append(img)
        
        return images

def read_mnist_label(filename: str, item_count: int) -> List[int]:
    labels = list()

    with open(filename, "rb") as fin:
        assert int.from_bytes(fin.read(4), byteorder="big") == 0x00000801
        assert (N := int.from_bytes(fin.read(4), byteorder="big")) == item_count

        for t in range(N):
            if t % 1000 == 0:
                print(f"read_mnist_label() index = {t}")

            label = int.from_bytes(fin.read(1), byteorder="big")
            labels.append(label)
        
        return labels


def processImages(download_path, upload_path):
    images = read_mnist_image(download_path, const.TRAIN_SIZE)
    with open(upload_path, 'wb') as fp:
        pickle.dump(images, fp)

def processLabels(download_path, upload_path):
    labels = read_mnist_label(download_path, const.TRAIN_SIZE)
    with open(upload_path, 'wb') as fp:
        pickle.dump(labels, fp)
