# Ames Housing Dataset
Build an ML regression model to predict real estate prices using the Ames Housing Dataset.

# Dataset
The Ames Housing Dataset contains features on 2930 houses and was compiled by Dean De Cock for use in data science education.

# Project Architecture

# Repository Structure
The exploratory data analysis is performed in the [eda.ipynb](https://github.com/rossrco/ames_housing_ml/blob/master/eda.ipynb) notebook. The actual training is performed in the `trainer` module.

The `trainer` module contains a training script. At training time, the `trainer` module is packaged as a docker container based on the instructions in the `Dockerfile`. It is then submitted for training on GCP's ML Engine.