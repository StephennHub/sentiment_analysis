from zipfile import ZipFile
import os
from tqdm import tqdm
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from joblib import Parallel, delayed
from config import ZIPPEDFILE, OLDNAME, NEWNAME, COLUMNNAME

zipped_file = ZIPPEDFILE
old_name = OLDNAME
new_name = NEWNAME
col_names = COLUMNNAME
content_regex = re.compile("[^a-zA-Z]")
nltk.download("stopwords")
letters = stopwords.words("english")

port_stem = PorterStemmer()
tqdm.pandas()


def unzipper(zipped_file):
    with ZipFile(zipped_file, "r") as zipfile:
        unzipped_file = zipfile.extractall()
        print("FIle has unzipped")

    if os.path.exists(new_name):
        print(f"{new_name} already exists")
    else:
        os.rename(old_name, new_name)
        print(f"name has change to -{new_name}-")
    return unzipped_file


def stem_single_content(content):
    words = content_regex.sub(" ", content).lower().split()
    stemmed_content = [
        port_stem.stem(word) for word in words if not word in stopwords.words("english")
    ]
    return " ".join(stemmed_content)


def stemming(text_list):
    return Parallel(n_jobs=1, backend="threading")(
        delayed(stem_single_content)(text) for text in text_list
    )


def main():
    unzipper(zipped_file)
    data = pd.read_csv("dataset.csv", names=col_names, encoding="ISO-8859-1")
    data.replace({"target": {4: 1}}, inplace=True)

    data["stem_data"] = data["text"].progress_apply(stem_single_content)
    data.to_csv("updated_data.csv", index=False)
    print("file has saved")
    print("Process had completed")


if __name__ == "__main__":
    main()
