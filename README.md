# TranSQ

MICCAI 22 accepted paper [TranSQ: Transformer-based Semantic Query for Medical Report Generation](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_58) presents a method for generating medical reports using semantic queries.

## Step 1: Pre-process

Run the following scripts in `./preprocess`: `data_preprocess.py`, `generate_CE.py`, and `generate_sentence_gallery.py` to obtain preprocessed data, semantic query category initialization, and build a sentence retrieval library.

## Step 2: Train/Evaluation

Modify the `transq/config.py` configuration file and run the `run.py` script (refer to `./model/README.md` for obtaining pre-trained models).

## Step 3: Post-process & Evaluate

1. Test the trained model on the training set to gather data for averaging the positional order of semantic vectors.
2. Run the `evaluation/NLG_eval/test_from_json.py` script to calculate the average positional order of the mentioned sentences in the training set results. Perform a re-ranking of the test set results and compute the final metrics.

Note: Since there are slight differences between IU X-ray and MIMIC-CXR (different image numbers), we implemented the two datasets with two separate projects for convenience, and the main branch is for MIMIC-CXR.
