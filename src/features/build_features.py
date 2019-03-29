# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
import ast
from src.features.w2v import train_w2v, document_vector


#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(data_dir):
    """ Runs data processing scripts to turn cleaned data from (../processed) into
        features ready to train models (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info('load gateway to research data')
    docs = (pd.read_csv(f"{data_dir}/processed/gtr_tokenised.csv",
            usecols=['processed_documents']).processed_documents.apply(ast.literal_eval))
    logger.info('making gateway to research word embeddings')
    w2v = train_w2v(docs)
    w2v.save(f"{project_dir}/models/gtr_w2v")

    logger.info('making gateway to research document vectors')
    doc_vecs = pd.DataFrame([document_vector(w2v, doc) for doc in docs], index=docs.index, columns=[f'dim_{i}' for i in range(w2v.vector_size)]).to_csv(f"{data_dir}/processed/gtr_embedding.csv")


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

    main(data_dir)
