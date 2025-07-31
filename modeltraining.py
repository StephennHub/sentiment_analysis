import os.path
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from config import FILENAME, FILEVECTOR

vectorizer = TfidfVectorizer()
logregModel = LogisticRegression(max_iter=1000)
filename = FILENAME
filevector = FILEVECTOR


def model_training(clean_data):
    x, y = clean_data["stem_data"].values, clean_data["target"].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=2
    )

    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    logregModel.fit(x_train_vec, y_train)
    x_train_prediction = logregModel.predict(x_test_vec)
    return x_train_prediction, y_test


def acc_score(x_train_prediction, y_test):
    test_accuracy = accuracy_score(y_test, x_train_prediction)
    return print(f"Accuracy Score is : {test_accuracy:.2f}%")


def training():
    updated_data = pd.read_csv("updated_data.csv")
    cleaned_data = updated_data.dropna(subset=["stem_data"])
    print("data updated and cleaned")

    x_train_prediction, y_test = model_training(cleaned_data)
    acc_score(x_train_prediction, y_test)
    print("process completed")

    if os.path.exists(filename) and os.path.exists(filevector):
        print(f"{filename} and {filevector} is already exists")
    else:
        with open(filename, "wb") as modelfile:
            pkl.dump(logregModel, modelfile)
            print(f"file saved in {filename}")

        with open(filevector, "wb") as vec_file:
            pkl.dump(vectorizer, vec_file)
            print(f"file saved in {filevector}")


if __name__ == "__main__":
    training()
