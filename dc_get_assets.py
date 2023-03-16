
import boto3
import botocore

import deepchecks.tabular as dct
import joblib
import pandas as pd

from pathlib import Path
import os.path as osp


BUCKET_NAME = 'deepchecks-public' 
BUCKET_KEY_BASE = 'datasets/titanic' 

TRAIN_FILENAME = "titanic_train.csv"
TEST_FILENAME = "titanic_test.csv"
MODEL_FILENAME = "titanic_rf.model"

OUTPUT_DATA_DIR = "example_data"

MODEL_FILE = Path(OUTPUT_DATA_DIR, MODEL_FILENAME)
TRAIN_FILE = Path(OUTPUT_DATA_DIR, TRAIN_FILENAME)
TEST_FILE = Path(OUTPUT_DATA_DIR).joinpath(TEST_FILENAME)



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


def get_train_ds():
    download_titanic_file(TRAIN_FILENAME)
    train_data = pd.read_csv(TRAIN_FILE)
    train_ds = dct.Dataset(train_data, **dataset_metadata)
    return train_ds



def get_test_ds():
    download_titanic_file(TEST_FILENAME)
    test_data = pd.read_csv(TEST_FILE)
    test_data['Survived'] = test_data['Survived'].sample(frac=1)
    test_ds = dct.Dataset(test_data, **dataset_metadata)
    return test_ds

# from sklearn.ensemble import RandomForestClassifier 

# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(new_train_ds.features_columns, new_train_ds.label_col)

# can either train here model and then save it and be able to predict, or just load a saved model
def load_model(train_dataset=None, **kwargs):
    download_titanic_file(MODEL_FILENAME)
    model = joblib.load(MODEL_FILE)
    return model



def main():
    pass


if __name__ == "__main__" :
    main()
