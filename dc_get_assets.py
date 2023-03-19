
import boto3
import botocore
import deepchecks.tabular as dct
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


######## File names ############
BUCKET_NAME = 'deepchecks-public' 
BUCKET_KEY_BASE = 'datasets/titanic' 
TRAIN_FILENAME = "titanic_train.csv"
TEST_FILENAME = "titanic_test.csv"
MODEL_FILENAME = "titanic_rf.model"
OUTPUT_DATA_DIR = "example_data"
MODEL_FILE = Path(OUTPUT_DATA_DIR, MODEL_FILENAME)
TRAIN_FILE = Path(OUTPUT_DATA_DIR, TRAIN_FILENAME)
TEST_FILE = Path(OUTPUT_DATA_DIR).joinpath(TEST_FILENAME)
###############################


dataset_metadata = {'cat_features' : ['Pclass', 'SibSp', 'Parch', 'Sex_male'],
                    'label':'Survived',
                    'label_type':'binary'}



def download_titanic_file(filename):
    
    # create folder for download if doesn't exist
    Path(OUTPUT_DATA_DIR).mkdir(parents=True, exist_ok=True)

    s3 = boto3.resource('s3')
    try:
        with open(Path(OUTPUT_DATA_DIR, filename), 'wb') as data:
            s3.Bucket(BUCKET_NAME).download_fileobj(str(Path(BUCKET_KEY_BASE, filename)), data)


    except botocore.exceptions.ClientError as e: 
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise e


def get_train_dataset():
    download_titanic_file(TRAIN_FILENAME)
    train_data = pd.read_csv(TRAIN_FILE)
    train_dataset = dct.Dataset(train_data, **dataset_metadata)
    return train_dataset


def get_test_dataset():
    download_titanic_file(TEST_FILENAME)
    test_data = pd.read_csv(TEST_FILE)
    # demonstrating that for optimization purposes,
    #DataFrame was converted to numpy was used for processing in pipeline,
    # and then inserted back to DataFrame    
    # test_data[dataset_metadata['label']] = np.array(
    #     test_data[dataset_metadata['label']].sample(frac=1))
    test_dataset = dct.Dataset(test_data, **dataset_metadata)
    return test_dataset


def load_model(train_dataset=None):
    download_titanic_file(MODEL_FILENAME)
    model = joblib.load(MODEL_FILE)
    return model



def main():
    pass


if __name__ == "__main__" :
    main()
