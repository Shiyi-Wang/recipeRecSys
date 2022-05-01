# Thursday -  A CS 4675 Project
Built with ❤️ by [Shiyi Wang](mailto:swang793@gatech.edu), [Shuangyue Cheng](mailto:katcheng@gatech.edu), [Taichang Zhou](mailto:tzhou915@gatech.edu), [Shuyan Lin](mailto:slin915@gatech.edu), [Haoran Zhao](mailto:hzhao353@gatech.edu)

## Table of Contents
- [Introduction](#Introduction)
- [Getting Started](#GettingStarted)
- [Backend Executables](#GettingStarted)
- [Final Executables](#GettingStarted)

## Introduction
We present Thursday, a Transparent Hybrid User-adjustable Recipe recommender System. The training dataset is [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv) from Kaggle.

## Getting Started

### 1. Environment Setup

1. Create the conda environment from `thursday.yml` in the index directory.

```sh
$ conda env create -f thursday.yml
```

2. To activate the environment, use
```sh
$ conda activate thursday
```

3. To deactivate an active environment, use
```sh
$ conda deactivate
```

### 2. Dataset Downloads

1. [Download from Google Drive](https://drive.google.com/drive/folders/1-f0rpDQ_XbWw9TOlKIIqJDdUqekTdSIC?usp=sharing).

2. Move all downloaded files to <code>data</code> directory.

3. Verify <code>data</code> directory has all required files.
```sh
.
├── SVD_algo.pkl
├── ingredients.json
├── interactions.csv
├── processed_data.pkl
├── recipes.csv
├── recipes_names.pkl
└── top_ingredients.json
```

## Backend Executables

### Directory Structure
```sh
.
├── README.md
├── app          # APIs
├── data         # Datasets
├── demo         # Demo output
├── documents    # Documentation files
├── evaluations  # Evaluation metrics
├── frontend     # Frontend assets
├── images       # Evaluation visualizations
├── models       # Backend RS models
├── outputs      # Jupyter Notebook snapshots
├── thursday.yml # conda environment setup file
└── tools        # Tools and utilities
```

### Model Module Executables

#### Runtime Notes

Runtime varies amomg machines. On Macbook Pro (2019) (2.4 GHz Quad-Core Intel Core i5):

* Collaborative Filtering with KNN - 40 minutes
* Collaborative Filtering with SVD - 15 minutes
* Content-based Filtering with NLTK - 20 minutes

#### Collaborative Filtering with KNN

Option 1: View [Jupyter Notebook HTML snapshot](https://github.com/Shiyi-Wang/recipeRecSys/blob/main/outputs/Collaborative%20Filtering%20with%20KNN.html).

Option 2: Run Jupyter Notebook.

In <code>models</code> directory, run the following command:

```sh
jupyter notebook Collaborative\ Filtering\ with\ KNN.ipynb 
```

#### Collaborative Filtering with SVD

Option 1: View [Jupyter Notebook HTML snapshot](https://github.com/Shiyi-Wang/recipeRecSys/blob/main/outputs/Collaborative%20Filtering%20with%20SVD.html).

Option 2: Run Jupyter Notebook.

In <code>models</code> directory, run the following command:

```sh
jupyter notebook Collaborative\ Filtering\ with\ SVD.ipynb 
```

#### Content-baed Filtering with NLTK

[Link to external Google Colab](https://colab.research.google.com/drive/1eq5x3gYnl_-8Rszju_L9TFl5tnpFkhB8)

> Note: modifications has been made on the API implementation 

### Evaluation Module Executables

Option 1: View Jupyter Notebook HTML snapshots.

* [Benchmark Evaluation.html](https://github.com/Shiyi-Wang/recipeRecSys/blob/main/outputs/Benchmark%20Evaluation.html)
* [Dataset Evaluation.html](https://github.com/Shiyi-Wang/recipeRecSys/blob/main/outputs/Dataset%20Evaluation.html)
* [KNN Evaluation.html](https://github.com/Shiyi-Wang/recipeRecSys/blob/main/outputs/KNN%20Evaluation.html)
* [SVD Evaluation.html](https://github.com/Shiyi-Wang/recipeRecSys/blob/main/outputs/SVD%20Evaluation.html)

Option 2: Run Jupyter Notebooks in <code>evaluations</code> directory.

## Final Executable
