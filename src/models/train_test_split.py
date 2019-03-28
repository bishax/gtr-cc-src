
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--train_size', type=float, default=0.8)
@click.option('--test_size', type=float, default=None)
@click.option('--random_state', type=int, default=0)
def main(train_size, test_size, random_state):
    """ Performs test-train split
    """
    logger = logging.getLogger(__name__)

    logger.info('load gateway to research data')
    Xy = (pd.read_csv(f"{data_dir}/processed/gtr_tokenised.csv",
            usecols=['funder_name'])
            .join(pd.read_csv(f"{data_dir}/processed/gtr_embedding.csv", index_col=0)
                )
            )

    target = 'funder_name'
    X_train, X_test, y_train, y_test = train_test_split(Xy.drop(target, 1), Xy[target], train_size=train_size, test_size=test_size, random_state=random_state, shuffle=True)

    df_train = X_train.join(y_train)
    df_test = X_test.join(y_test)

    df_train.to_csv(f"{data_dir}/processed/gtr_train.csv")
    df_test.to_csv(f"{data_dir}/processed/gtr_test.csv")


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
