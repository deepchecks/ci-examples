import deepchecks.tabular as dct
from deepchecks.tabular.suites import data_integrity
import test_suites
import sys 

# append to path parent folder in order to import
import os 
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(dir_path)
sys.path.append(dir_path)
import dc_get_assets


def run_suite(suite, run_args):
    if type(suite) != dct.Suite:
        raise ValueError("Expected object of type Suite, received: {}".format(type(suite)))
    result = suite.run(**run_args)
    return result


def load_assets():
    train_ds = dc_get_assets.get_train_ds()
    test_ds = dc_get_assets.get_test_ds()
    model_obj = dc_get_assets.load_model()
    return train_ds, test_ds, model_obj


def load_and_run(suites_to_run):
    train_ds, test_ds, model_obj = load_assets()
    suite_args = {"train_dataset": train_ds, "test_dataset": test_ds, "model": model_obj}
    

    results = []
    if type(suites_to_run) == list:
        for suite in suites_to_run:
            results.append(run_suite(suite, suite_args))
    else:
        results.append(run_suite(suite, **suite_args))
    return results



def main():
    manual_suites_selection = [test_suites.first_custom_suite() ,test_suites.my_model_evaluation(), data_integrity()]
    results=load_and_run(manual_suites_selection)
    return results


if __name__ == "__main__":
    main()