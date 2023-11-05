import pandas as pd
from datasets import Dataset, DatasetDict
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


download_and_unzip("https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip",
                   extract_to='../../data/raw')

df = pd.read_table("../../data/raw/filtered.tsv")

toxics = df["reference"].values.tolist()
neutrals = df["translation"].values.tolist()

toxics = list(map(lambda x: x.lower().strip(), toxics))
neutrals = list(map(lambda x: x.lower().strip(), neutrals))

train, val, test = 0.7, 0.1, 0.2
n = len(toxics)

toxic_sent = "toxic_comment"
target_sent = "neutral_comment"

train_data = {
    toxic_sent: toxics[:int(n * train)],
    target_sent: neutrals[:int(n * train)]
}

train_dataset = Dataset.from_dict(train_data)

val_data = {
    toxic_sent: toxics[int(n * train): int(n * (train + val))],
    target_sent: neutrals[int(n * train): int(n * (train + val))]
}

val_dataset = Dataset.from_dict(val_data)

test_data = {
    toxic_sent: toxics[int(n * (train + val)):],
    target_sent: neutrals[int(n * (train + val)):]
}

test_dataset = Dataset.from_dict(test_data)

dataset = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})

dataset_path = "../../data/interim/"
dataset.save_to_disk(dataset_path)
