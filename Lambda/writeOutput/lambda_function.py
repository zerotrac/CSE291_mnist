import json
import numpy as np
import cv2
import boto3
import pickle
from typing import List

import const

s3_client = boto3.client('s3')

class Box:
    def __init__(self, wmin: int, hmin: int, wmax: int, hmax: int, compressed_img: List[int]):
        self.wmin = wmin
        self.hmin = hmin
        self.wmax = wmax
        self.hmax = hmax
        self.compressed_img = compressed_img

def collect_result(filename: str, output_name: str, boxes: List[Box], predictions: List[int]):
    img = cv2.imread(filename)
    h, w, _ = img.shape
    
    for box, prediction in zip(boxes, predictions):
        img = cv2.rectangle(img, (box.wmin, box.hmin), (box.wmax, box.hmax), (87, 201, 0), 2)
        img = cv2.putText(img, str(prediction), ((box.wmin + box.wmax) // 2, box.hmin), cv2.FONT_HERSHEY_COMPLEX, min(h, w) / 500, (255, 144, 30), 2)
    
    cv2.imwrite(output_name, img)


# image to boxes
def lambda_handler(event, context):
    for record in event["Records"]:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        download_path = '/tmp/{}'.format(key)
        upload_path = '/tmp/ocr_{}'.format(key)
        s3_client.download_file(bucket, key, download_path)
        
        
        # get predictions
        with open (download_path, 'rb') as fp:
            predictions = pickle.load(fp)
        print("predictions {} downloaded from {}".format(key ,bucket))
        # get original image
        # original_bucket = 'mnist-prediction'
        original_download_path = '/tmp/raw_{}'.format(key)
        s3_client.download_file(const.S3original, key, original_download_path)
        print("original image {} downloaded from {}".format(key ,const.S3original))
        # get boxes
        # boxes_bucket = 'mnist-boxes'
        boxes_download_path = '/tmp/boxes_{}'.format(key)
        s3_client.download_file(const.S3boxes, key, boxes_download_path)
        with open (boxes_download_path, 'rb') as fp:
            boxes = pickle.load(fp)
        print("boxes {} downloaded from {}".format(key ,const.S3boxes))
        # draw output with boxes
        collect_result(original_download_path, upload_path, boxes, predictions)
        print("draw output done")
        
        s3_client.upload_file(upload_path, const.S3output, key)
        print("new output image uploaded to {}".format(const.S3output))
        
    return {
        'statusCode': 200,
        'body': json.dumps('done!')
    }

