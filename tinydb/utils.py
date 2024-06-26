import os
import numpy
import shutil
import tarfile
import urllib.request as request

from contextlib import closing



DATA_PATH = os.path.join("data", "siftsmall", "siftsmall_base.fvecs")
QUERY_PATH = os.path.join("data", "siftsmall", "siftsmall_query.fvecs")
LABEL_PATH = os.path.join("data", "siftsmall", "siftsmall_groundtruth.ivecs")


def download_sift():
    output= os.path.join("data", "siftsmall.tar.gz")
    with closing(
        request.urlopen("ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz")
    ) as r:
        with open(output, "wb") as f:
            shutil.copyfileobj(r, f)
    tar = tarfile.open(output, "r:gz")
    tar.extractall("data")

def read_vecs(path: str, ivecs: bool = False) -> numpy.ndarray:
    a = numpy.fromfile(path, dtype="int32")
    d = a[0]
    matrix = a.reshape(-1, d + 1)[:, 1:].copy()

    if not ivecs:
        matrix = matrix.view("float32")

    return matrix
def evaluate(gold: numpy.ndarray, predictions: numpy.ndarray) -> float:
    """
    Compute Recall@1;
        - gold: array of shape (k,) -- integers
        - predictions: array of shape (k,) -- integers
    """
    return sum(gold==predictions)