# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score


@click.command()
@click.option('--random_state', type=int, default=0)
def main(random_state):
    """ Runs model
    """
    logger = logging.getLogger(__name__)

    Xy = pd.read_csv(f"{data_dir}/processed/gtr_test.csv", index_col=0)
    X, y = Xy.drop('funder_name', 1), Xy.funder_name

    clf = joblib.load(f"{project_dir}/models/gtr_forest.pkl")

    accuracy = accuracy_score(y, clf.predict(X))
    logging.info(f"Test Accuracy: {accuracy}")

    with open(f"{project_dir}/models/metrics.txt", 'w') as f:
        f.write(f"{'gtr_clf'} accuracy: {accuracy:4f}\n")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    data_dir = project_dir / 'data'
    print(data_dir)

    main()
