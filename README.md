# Ames Housing Dataset
Build an ML regression model to predict real estate prices using the Ames Housing Dataset. In this project, we use a dataset and perform a rapid analysis, followed by minimal data transformation. The goal of the project is to place as accurate model as possible as quickly as possible.

# Dataset
The Ames Housing Dataset contains features on 2930 houses and was compiled by Dean De Cock for use in data science education.

# Project Architecture
The train dataset is transformed using `tensorflow_transform`. The result is a transformed train dataset as well as a transformation function. The transformation function is used together with the model at serve time.

# Repository Structure
The exploratory data analysis is performed in the [eda.ipynb](https://github.com/rossrco/ames_housing_ml/blob/master/eda.ipynb) notebook. The actual training is performed in the `trainer` module.

The `prototype.ipynb` notebook contains an initial proof of concept of the transform functions.

The `trainer` module contains a training script. At training time, the `trainer` module is packaged as a docker container based on the instructions in the `Dockerfile`. It is then submitted for training on GCP's ML Engine.