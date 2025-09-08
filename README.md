# Small Language Models Exploration

## Introduction
This project serves as an investigation on the effectiveness of vanilla RNNs and LSTMs, as part of an assignment for DSA4213 (Natural Language Processing for Data Science).

## Setup Instructions
1. Ensure that you have Python 3.13 installed. If not, you can visit the [Python website](https://www.python.org/downloads/) for instructions on installation. Once installed, you can verify your version of Python by running the following in your terminal:

```
python --version
```

2. If you do not have Git installed, visit the [Git website](https://git-scm.com/downloads) for instructions on installation. Once installed, you can verify your version of Git by running the following in your terminal:

```
git --version
```

3. Clone the repository as follows:
```
git clone https://github.com/chiabingxuan/Small-Language-Models-Comparison.git
```

4. Set your working directory to the folder containing the cloned repository:
```
cd Small-Language-Models-Comparison
```

5. Create a Python virtual environment named `venv/`:

```
python -m venv venv
```

6. Activate the virtual environment:

```
venv\Scripts\activate
```

7. Install necessary packages:
```
pip install -r requirements.txt
```

8. To obtain, process and save the data from the Reuters corpus, run `data_processing.ipynb`. Under "Getting the Data (Reuters)", you can adjust the train-validation-test split to your liking. The tokenised data will be saved under `data/` directory.

9. To carry out model training, run `model_training.ipynb`. Under "Set Model Hyperparameters" and "Set Training Hyperparameters", you can adjust the respective hyperparameters to your liking. The `state_dict` of the trained models will be saved under `weights/` directory. Plots of the loss curves will be saved under `plots/` directory.

10. To evaluate model performance, run `model_evaluation.ipynb`. Under "Set Parameters", make sure the parameter values correspond to those that were set during model training.