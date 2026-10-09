"""Download the public Kaggle dataset without requiring private credentials."""
from pathlib import Path
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
URL='https://www.kaggle.com/api/v1/datasets/download/julian3833/jigsaw-toxic-comment-classification-challenge'
EXPECTED_SHA256='bd4084611bd27c939ba98e5e63bc3e5a2c1a4e99477dcba46c829e4c986c429d'


def download():
    output=ROOT/'data/raw/train.csv'
    if output.exists():
        digest=hashlib.sha256(output.read_bytes()).hexdigest()
        if digest==EXPECTED_SHA256:
            print('Existing original training CSV verified.'); return
        raise ValueError('Existing CSV has a different checksum; preserve it and investigate before downloading.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'jigsaw.zip'
        request=urllib.request.Request(URL,headers={'User-Agent':'Neurova-NLP-course-project'})
        with urllib.request.urlopen(request,timeout=180) as source, archive.open('wb') as target:
            shutil.copyfileobj(source,target)
        with zipfile.ZipFile(archive) as files:
            with files.open('train.csv') as source, output.open('wb') as target:
                shutil.copyfileobj(source,target)
    if hashlib.sha256(output.read_bytes()).hexdigest()!=EXPECTED_SHA256:
        raise ValueError('Downloaded file differs from the original recorded CSV. Review the dataset version.')
    print('Original Jigsaw training CSV downloaded and verified.')


if __name__=='__main__':
    download()
