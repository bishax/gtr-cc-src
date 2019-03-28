import pandas as pd
from src.features.text_preprocessing import tokenize_document

def make_gtr(data_dir):
    """
    """

    gtr_df = (pd.read_csv(f"{data_dir}/raw/gtr_projects.csv", nrows=1000)
            .pipe(clean_gtr)
            .pipe(transform_gtr)
            .to_csv(f"{data_dir}/processed/gtr_tokenised.csv")
            )


def clean_gtr(gtr_df):
    """
    """

    return (gtr_df
            .drop_duplicates('project_id')
            .dropna()
            )


def transform_gtr(gtr_df):
    """
    """

    processed = (gtr_df.abstract_texts
            .apply(tokenize_document, flatten=True)
            .to_frame('processed_documents')
            .assign(is_doc=lambda x: x.processed_documents.apply(len) > 0)
            .query("is_doc > 0")
                )

    return gtr_df.join(processed)

