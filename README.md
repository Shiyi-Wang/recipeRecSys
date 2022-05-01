# Thursday: Transparent Hybrid User-adjustable Recipe RS -  A CS 4675/6675 Project
Contributors: [Shiyi Wang](mailto:swang793@gatech.edu), [Shuangyue Cheng](mailto:katcheng@gatech.edu), [Taichang Zhou](mailto:tzhou915@gatech.edu), [Shuyan Lin](mailto:slin915@gatech.edu), [Haoran Zhao](mailto:hzhao353@gatech.edu)

## Introduction
We present Thursday, a Transparent Hybrid User-adjustable Recipe recommender System. The training dataset is available at [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv) from Kaggle.

## Folder Structure
```zsh
.
├── README.md
├── app
│   ├── Content.py
│   ├── MeanEmbeddingVectorizer.py
│   ├── TfidfEmbeddingVectorizer.py
│   ├── add_new_user.py
│   ├── demo.py
│   ├── hybrid_api.py
│   ├── knn_api.py
│   ├── model_cbow.bin
│   └── svd_api.py
├── data
│   ├── SVD_algo.pkl
│   ├── ingredients.json
│   ├── interactions.csv
│   ├── processed_data.pkl
│   ├── recipes.csv
│   ├── recipes_names.pkl
│   └── top_ingredients.json
├── demo
│   ├── avg_case_demo.json
│   ├── easy_case_demo.json
│   └── hard_case_demo.json
├── documents
│   ├── Demo Presentation.pdf
│   ├── Proposal.pdf
│   └── Workshop Presentation.pdf
├── evaluations
│   ├── Benchmark Evaluation.ipynb
│   ├── Dataset Evaluation.ipynb
│   ├── KNN Evaluation.ipynb
│   └── SVD Evaluation.ipynb
├── frontend
│   └── test.js
├── images
│   ├── KNN Classification Probability.png
│   ├── KNN Coverage.png
│   ├── KNN Mark.png
│   ├── KNN Precision Recall Curve.png
│   ├── KNN ROC AUC.png
│   ├── KNN vs. SVD vs. Hybrid.png
│   ├── Long Tail Plot.png
│   ├── SVD Classification Probability.png
│   ├── SVD Coverage.png
│   ├── SVD Mark.png
│   ├── SVD Precision Recall Curve.png
│   └── SVD ROC AUC.png
├── models
│   ├── Collaborative Filtering with KNN.ipynb
│   ├── Collaborative Filtering with SVD.ipynb
│   └── Preprocessing.ipynb
├── outputs
│   ├── Benchmark Evaluation.html
│   ├── Collaborative Filtering with KNN.html
│   ├── Collaborative Filtering with SVD.html
│   ├── Content-based Filtering with NLTK.html
│   ├── Dataset Evaluation.html
│   ├── KNN Evaluation.html
│   ├── Preprocessing.html
│   └── SVD Evaluation.html
├── thursday.yml
└── tools
    ├── AddNewUser.ipynb
    └── RecipeLookUp.ipynb
```

## Getting Started

### Environment Setup

1. Create the conda environment from `thursday.yml` in the index directory.

```zsh
$ conda env create -f thursday.yml
```

2. To activate the environment, use
```zsh
$ conda activate thursday
```

3. To deactivate an active environment, use
```zsh
$ conda deactivate
```

### File Downloads

##

### Models

### Evaluations

## Final Executables
