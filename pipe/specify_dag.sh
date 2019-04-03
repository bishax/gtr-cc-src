#!/bin/bash

set -e

# Clean and tokenise
dvc run -w ..\
  -d data/raw/gtr_projects.csv\
  -d src/features/text_preprocessing.py\
  -o data/processed/gtr_tokenised.csv\
  python src/data/make_dataset.py -n 10

# Train word embeddings
dvc run -w .. \
  -d data/processed/gtr_tokenised.csv\
  -d src/features/w2v.py\
  -o models/gtr_w2v\
  -o data/processed/gtr_embedding.csv\
  python src/features/build_features.py

# Test-train split
dvc run -w ..\
  -d data/processed/gtr_embedding.csv\
  -d data/processed/gtr_tokenised.csv\
  -o data/processed/gtr_train.csv\
  -o data/processed/gtr_test.csv\
  python src/models/train_test_split.py

# Train model
dvc run -w ..\
  -d data/processed/gtr_train.csv\
  -o models/gtr_forest.pkl\
  python src/models/predict_model.py

# Evaluate model
touch ../models/metrics.txt
dvc run -w ..\
  -d data/processed/gtr_test.csv\
  -d models/gtr_forest.pkl\
  -M models/metrics.txt\
  -f Dvcfile
  python src/models/evaluate.py
