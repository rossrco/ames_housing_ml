from __future__ import print_function
from io import BytesIO

from google.cloud import storage

import pandas as pd

import argparse


def train(**args):
    print('Ingesting data.')
    client = storage.Client()
    bucket = client.get_bucket('ames-house-dataset')
    blob = storage.Blob('train.csv', bucket)
    content = blob.download_as_string()
    data = pd.read_csv(BytesIO(content), index_col=0)
    print('Data extracted.')
    print(data.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    arguments = args.__dict__
    train(**arguments)
