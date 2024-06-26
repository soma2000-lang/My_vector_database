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