import dc_get_assets
import deepchecks.tabular as dct
import test_suites


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


def load_and_run():
    dc_get_assets.download_titanic_files()
    train_ds, test_ds, model_obj = load_assets()
    suite_args = {"train_dataset": train_ds, "test_dataset": test_ds, "model": model_obj}

    # this is manual...
    suites_to_run = [test_suites.first_custom_suite() ,test_suites.my_model_evaluation()]
    

    results = []
    if type(suites_to_run) == list:
        for suite in suites_to_run:
            results.append(run_suite(suite, suite_args))
    else:
        results.append(run_suite(suite, **suite_args))
    return results




def main():
    print("run load_and_run() to execute")

if __name__ == "main":
    main()


load_and_run()