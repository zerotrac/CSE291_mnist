import json
import numpy as np
import cv2
import boto3
import pickle

import const

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    for record in event["Records"]:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        download_path = '/tmp/{}'.format(key)
        upload_path = '/tmp/processed-{}'.format(key)
        s3_client.download_file(bucket, key, download_path)
        print("mnist input {} downloaded from {}".format(key ,bucket))

        predictInput(download_path, upload_path)

        s3_client.upload_file(upload_path, const.S3output, key)
        print("new picture uploaded to {}".format(const.S3output))
