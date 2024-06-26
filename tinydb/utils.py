import os
import numpy
import shutil
import tarfile
import urllib.request as request

from contextlib import closing

from contextlib import closing


DATA_PATH = os.path.join("data", "siftsmall", "siftsmall_base.fvecs")
QUERY_PATH = os.path.join("data", "siftsmall", "siftsmall_query.fvecs")
LABEL_PATH = os.path.join("data", "siftsmall", "siftsmall_groundtruth.ivecs")


def read_vecs(path: str, ivecs: bool = False) -> numpy.ndarray:
    a = numpy.fromfile(path, dtype="int32")
    d = a[0]
    matrix = a.reshape(-1, d + 1)[:, 1:].copy()

    if not ivecs:
        matrix = matrix.view("float32")

    return matrix