# TranSQ

MICCAI 22 accepted paper [TranSQ: Transformer-based Semantic Query for Medical Report Generation](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_58) presents a method for generating medical reports using semantic queries.

## Step 1: Pre-process

Run the following scripts in `./preprocess`: `data_preprocess.py`, `generate_CE.py`, and `generate_sentence_gallery.py` to obtain preprocessed data, semantic query category initialization, and build a sentence retrieval library.

## Step 2: Train/Evaluation

Modify the `transq/config.py` configuration file and run the `run.py` script (refer to `./model/README.md` for obtaining pre-trained models). For example:

'''
python run.py with train_mimic_vit
'''

## Step 3: Post-process & Evaluate

1. Test the trained model on the training set to gather data for averaging the positional order of semantic vectors.
2. Run the `evaluation/NLG_eval/test_from_json.py` script to calculate the average positional order of the mentioned sentences in the training set results. Perform a re-ranking of the test set results and compute the final metrics.

Note: Since there are slight differences between IU X-ray and MIMIC-CXR (different image numbers), we implemented the two datasets with two separate projects for convenience, and the main branch is for MIMIC-CXR.

# Acknowledgment:
The code implementation is modified from the project: https://github.com/dandelin/ViLT

# Contact:
If you have any problem, please feel free to contact me at zjukongming@zju.edu.cn

# Citation
If you use any part of this code and pre-trained weights for your own purpose, please cite our paper.
```
@article{GAO2024102982,
title = {Simulating doctors’ thinking logic for chest X-ray report generation via Transformer-based Semantic Query learning},
journal = {Medical Image Analysis},
volume = {91},
pages = {102982},
year = {2024},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2023.102982},
url = {https://www.sciencedirect.com/science/article/pii/S1361841523002426},
author = {Danyang Gao and Ming Kong and Yongrui Zhao and Jing Huang and Zhengxing Huang and Kun Kuang and Fei Wu and Qiang Zhu},
keywords = {Medical report generation, Semantic query, Transformer, Computer-aided diagnosis, Deep learning}
}
```

or

```
@InProceedings{10.1007/978-3-031-16452-1_58,
author="Kong, Ming
and Huang, Zhengxing
and Kuang, Kun
and Zhu, Qiang
and Wu, Fei",
editor="Wang, Linwei
and Dou, Qi
and Fletcher, P. Thomas
and Speidel, Stefanie
and Li, Shuo",
title="TranSQ: Transformer-Based Semantic Query for Medical Report Generation",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="610--620"
}
```
