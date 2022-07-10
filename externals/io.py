from typing import Dict, Callable, Union, Any

import warnings

import os

import copy

import pandas as pd

from tqdm import tqdm

import tensorflow as tf


__all__ = ['TFRecordWriter', 'TFRecordReader']


def split(size: int, n: int):

    chunks = size // n

    if size % n:

        chunks += 1

    for i in range(n):

        start = chunks * i

        end = chunks * (i + 1)
        end = min(end, size)

        yield start, end


class TFRecordWriter:

    def __init__(self, n_records: int,  failure_ignore: bool = False, cache_warnings: bool = False):

        """
        Parameters
        ----------

        n_records: int

        failure_ignore: bool
            if True, then Get a warning message in case of
            failed reading an image file or writing any feature of an example,
            instead of raising ValueError,
            e.g., skip reading a corrupted files and continue writing TFRecord,

            default = False

        cache_warnings: bool
            if True, cache warnings messages, as dict,
            index of the failure example as a key, and causes  (error: failure message) as value,

            cache, failure_examples attribute

            default = False
        """

        self.n_records = n_records
        self.failure_ignore = failure_ignore
        self.cache_warnings = cache_warnings
        self.failure_examples = None

    @staticmethod
    def _bytes_feature(value):

        """Returns a bytes_list from a string / byte."""

        # BytesList won't unpack a string from an EagerTensor.
        if hasattr(value, 'numpy'):

            value = value.numpy()

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):

        """Returns a float_list from a float / double."""

        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):

        """Returns an int64_list from a bool / enum / int / uint."""

        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _as_feature_type(feature_dtype, value):

        if feature_dtype == 'int':

            return TFRecordWriter._int64_feature(value)

        elif feature_dtype == 'float':

            return TFRecordWriter._float_feature(value)

        elif feature_dtype == 'bytes':

            return TFRecordWriter._bytes_feature(value)

        elif feature_dtype == 'str':

            return TFRecordWriter._bytes_feature(value.encode())

        else:

            raise ValueError('Unknown feature type, dtype=' + feature_dtype + ', is not supported')

    @staticmethod
    def preprocessing(row: Dict[Any, Any], row_dtypes: Dict[Any, Any]):

        """ convert media arrays to bytes """

        for key, dtype in row_dtypes.items():

            if dtype == 'media':

                row[key] = row[key].tobytes(order=None)
                row_dtypes[key] = 'bytes'

        return row, row_dtypes

    def parse_example(self, index: Any, features: Dict[Any, Any], features_dtypes: Dict[Any, Any],
                      from_dir: str = None, read_fn: Union[Callable, Dict[Any, Callable]] = None):

        row_dtypes = copy.deepcopy(features_dtypes)
        row = copy.deepcopy(features)

        if (read_fn is None) and ('media' in row_dtypes.values()):

            raise ValueError('Expected: read_fn: Callable or Dict[Any, Callable], found: None')

        for key in row:

            if row_dtypes[key] != 'media':

                continue

            if from_dir is None:

                path = row[key]

            else:

                path = os.path.join(from_dir, row[key])

            try:

                if isinstance(read_fn, dict):

                    value = read_fn[key](path)

                else:

                    value = read_fn(path)

            except Exception as error:

                if self.failure_ignore is True:

                    if self.cache_warnings is True:

                        self.failure_examples[index] = repr(error)

                    else:

                        warnings.warn(f'Warning: {repr(error)}')

                    return None, None

                else:

                    raise ValueError(f'An error has occurred, Failed parsing an example:'
                                     f'\n, {error}, to ignore failure state:\n'
                                     f'\tset TfRecordWriter(...., failure_ignore=True)')

            else:

                row[key] = value

        return row, row_dtypes

    def serialize_example(self, row: Dict[Any, Any], row_dtypes: Dict[Any, Any]):

        row, row_dtypes = self.preprocessing(row, row_dtypes)

        example = {}

        for key, value in row.items():

            example[key] = self._as_feature_type(row_dtypes[key], value)

        example = tf.train.Example(features=tf.train.Features(feature=example))

        return example.SerializeToString()

    def from_dataframe(self, dataframe: pd.DataFrame, dtypes: Union[list, dict], prefix: str = 'train',
                       from_dir: str = None, working_dir='./', read_fn: Union[Callable, Dict[Any, Callable]] = None):

        """
        Parameters
        ----------

        dataframe: pd.DataFrame

        dtypes: Union[list, dict]
            Encoding type for each dataframe column,
            utils primitive types, [bytes, int, float, str], image column must be equal (bytes),
            dtypes could be ordered list or dictionary of column name as a key and type as a value,
            type could be as string or type instance object, ex. {'image': 'bytes'} or {'image': bytes}

        prefix: str
            TfRecord file name prefix, pref_fname='train' - (ex. 'train_*.tfrec')

        from_dir: str
            media directory, default = None

        working_dir: str
            writing directory

        read_fn: Union[Callable, Dict[Any, Callable]]

        Returns
        -------
        """

        size = len(dataframe)

        for i, (start, end) in enumerate(split(size, self.n_records)):

            tfrecord_path = f'{working_dir}/{prefix}_{i}.tfrec'

            with tf.io.TFRecordWriter(tfrecord_path) as writer:

                progress_bar = tqdm(range(start, end))

                for j in progress_bar:

                    row, row_dtypes = self.parse_example(index=j, features=dataframe.iloc[j].to_dict(),
                                                         features_dtypes=dtypes, from_dir=from_dir, read_fn=read_fn)

                    if row is not None:

                        example = self.serialize_example(row=row, row_dtypes=row_dtypes)

                        writer.write(example)

                    if self.cache_warnings is True:

                        progress_bar.set_description('failure count = ' + str(len(self.failure_examples)))


class TFRecordReader:

    def __init__(self, dtypes: Dict[Any, Any], decode_fn: Dict[Any, Callable] = None):

        """
        dtypes: Dict[Any, Any]
            TfRecord features types, features_dtype is a dictionary of column name as a key and type as a value,
            e.g., type in ['int8', 'int16', ..., 'float16', 'float32', ..., 'str', 'bytes']

        decode_fn: Dict[Any, Callable]
        """

        self.dtypes = dtypes

        self.decode_fn = decode_fn

    @staticmethod
    def _feature_type(dtype):

        if dtype == 'int8':

            return tf.io.FixedLenFeature([], tf.int8)

        elif dtype == 'int16':

            return tf.io.FixedLenFeature([], tf.int16)

        elif dtype == 'int32':

            return tf.io.FixedLenFeature([], tf.int32)

        elif dtype == 'int64':
            return tf.io.FixedLenFeature([], tf.int64)

        elif dtype == 'float16':

            return tf.io.FixedLenFeature([], tf.float16)

        elif dtype == 'float32':

            return tf.io.FixedLenFeature([], tf.float32)

        elif dtype == 'float64':

            return tf.io.FixedLenFeature([], tf.float64)

        elif dtype in ['string', 'bytes']:

            return tf.io.FixedLenFeature([], tf.string)

        else:

            raise ValueError('Unknown type, dtype=' + dtype + ', is not supported')

    def read_example(self, example):

        """Parse features for a given `example`.
        """

        features = {}

        for key in self.dtypes:

            features[key] = TFRecordReader._feature_type(self.dtypes[key])

        # parser
        example = tf.io.parse_single_example(example, features)

        keys = list(features.keys())
        values = [None] * len(features)

        for i in range(len(keys)):

            key = keys[i]

            values[i] = example[key]

            if (self.decode_fn is not None) and (key in self.decode_fn):

                values[i] = self.decode_fn[key](values[i])

        return tuple(values)
