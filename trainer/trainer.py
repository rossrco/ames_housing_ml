from __future__ import print_function
from io import BytesIO
from google.cloud import storage
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
import argparse


def get_raw_data_spec(df):
    d_with_nulls = df.dtypes[df.isna().sum() > 0]
    d_no_nulls = df.dtypes[df.isna().sum() == 0]

    d_with_nulls_cat = d_with_nulls[d_with_nulls == np.dtype('O')]
    d_with_nulls_num = d_with_nulls[d_with_nulls.isin([np.dtype('int64'),
                                                       np.dtype('float64')])]

    d_no_nulls_cat = d_no_nulls[d_no_nulls == np.dtype('O')]
    d_no_nulls_num = d_no_nulls[d_no_nulls.isin([np.dtype('int64'),
                                                 np.dtype('float64')])]

    OPT_CAT_FEATURES = d_with_nulls_cat.index.to_list()
    OPT_NUM_FEATURES = d_with_nulls_num.index.to_list()
    CAT_FEATURES = d_no_nulls_cat.index.to_list()
    NUM_FEATURES = d_no_nulls_num.index.to_list()
    TARGET = 'SalePrice'

    RAW_DATA_FEATURE_SPEC = dict(
        [(name, tf.io.FixedLenFeature([], tf.string))
         for name in CAT_FEATURES]
        + [(name, tf.io.FixedLenFeature([], tf.float32))
           for name in NUM_FEATURES]
        + [(name, tf.io.VarLenFeature(tf.string))
           for name in OPT_CAT_FEATURES]
        + [(name, tf.io.VarLenFeature(tf.float32))
           for name in OPT_NUM_FEATURES]
        + [(TARGET, tf.io.VarLenFeature(tf.float32))])
    return RAW_DATA_FEATURE_SPEC


def train(**args):
    print('Ingesting data.')
    client = storage.Client()
    bucket = client.get_bucket('ames-house-dataset')
    blob = storage.Blob('train.csv', bucket)
    content = blob.download_as_string()
    data = pd.read_csv(BytesIO(content), index_col=0)

    print('Creating metadata specification.')
    RAW_DATA_FEATURE_SPEC = get_raw_data_spec(data)
    RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(RAW_DATA_FEATURE_SPEC))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    arguments = args.__dict__
    train(**arguments)
