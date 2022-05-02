# Thursday -  A CS 4675 Project
Built with passion by [Shiyi Wang](mailto:swang793@gatech.edu), [Shuangyue Cheng](mailto:katcheng@gatech.edu), [Taichang Zhou](mailto:tzhou915@gatech.edu), [Shuyan Lin](mailto:slin915@gatech.edu), [Haoran Zhao](mailto:hzhao353@gatech.edu)

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Backend Executables](#backend-executables)
- [Final Executable](#final-executable)

## Introduction
We present Thursday, a Transparent Hybrid User-adjustable Recipe recommender System.

This project offers the following features to address the state-of-the-art recommender system problems:

1. It utlizes Content-based Collaborative Filtering (CBCF) technique to offer optimized recommendation choices.
2. It leverages algorithm transparency and flexibility with user-adjustable weights.

The training dataset is [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv) from Kaggle.

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
3.seting up

```sh
$ pip install -e .
```

4. To deactivate an active environment, use
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
├── SVD_ratings.pkl
├── ingredients.json
├── interactions.csv
├── processed_data.pkl
├── recipes.csv
├── recipes_names.pkl
├── top_ingredients.json
├── id_ingredients.pkl
├── id_name.pkl
├── id_steps.pkl
└── RAW_recipes.csv
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

> Note: Modifications has been made on the final API.

### Evaluation Module Executables

Option 1: View Jupyter Notebook HTML snapshots.

* [Benchmark Evaluation.html](https://github.com/Shiyi-Wang/recipeRecSys/blob/main/outputs/Benchmark%20Evaluation.html)
* [Dataset Evaluation.html](https://github.com/Shiyi-Wang/recipeRecSys/blob/main/outputs/Dataset%20Evaluation.html)
* [KNN Evaluation.html](https://github.com/Shiyi-Wang/recipeRecSys/blob/main/outputs/KNN%20Evaluation.html)
* [SVD Evaluation.html](https://github.com/Shiyi-Wang/recipeRecSys/blob/main/outputs/SVD%20Evaluation.html)

Option 2: Run Jupyter Notebooks in <code>evaluations</code> directory.

### Hybrid API Demo
In <code>demo</code>, run <code>demo.py</code>
> Note: demo.py has a default <code>easy_case</code> demo as called in <code>main</code>.

You may switch with other cases defined in <code>demo.py</code>.

```sh
cd demo
python3 demo.py
```

## Final Executable
1. First run backend server by

```sh
cd recommender-server
python3 recipe.py
```

2. In  <code>frontend</code> directory, install [npm](https://www.npmjs.com/) by 

```sh
npm install
```

3. Open the react app by 

```sh
npm start
```
