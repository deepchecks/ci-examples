
import boto3
import botocore

import deepchecks.tabular as dct
import joblib
import pandas as pd

from pathlib import Path
import os.path as osp


BUCKET_NAME = 'deepchecks-public' 
PATH = 'datasets/titanic' 

TRAIN_FILENAME = "titanic_train.csv"
TEST_FILENAME = "titanic_test.csv"
MODEL_FILENAME = "titanic_rf.model"

OUTPUT_DATA_DIR = "example_data"

MODEL_FILE = Path(OUTPUT_DATA_DIR, MODEL_FILENAME)
TRAIN_FILE = Path(OUTPUT_DATA_DIR, TRAIN_FILENAME)
TEST_FILE = Path(OUTPUT_DATA_DIR).joinpath(TEST_FILENAME)



dataset_metadata = {'cat_features' : ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'],
                    'index_name': 'PassengerId',
                    'label':'Survived',
                    'label_type':'binary'}


def download_titanic_files():

    s3 = boto3.resource('s3')

    files_to_download = [TRAIN_FILENAME, TEST_FILENAME, MODEL_FILENAME]

    for filename in files_to_download:
        try:
            with open(osp.join(OUTPUT_DATA_DIR, filename), 'wb') as data:
                s3.Bucket(BUCKET_NAME).download_fileobj(osp.join(PATH, filename), data)


        except botocore.exceptions.ClientError as e: 
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise e


def get_train_ds():
    train_data = pd.read_csv(TRAIN_FILE)
    train_ds = dct.Dataset(train_data, **dataset_metadata)
    return train_ds


def get_test_ds():
    test_data = pd.read_csv(TEST_FILE)
    test_ds = dct.Dataset(test_data, **dataset_metadata)
    return test_ds


# can either train here model and then save it and be able to predict, or just load a saved model
def load_model():
    model = joblib.load(MODEL_FILE)
    return model



def main():
    pass


if __name__ == "__main__" :
    main()
