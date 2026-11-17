# Fraud Detection with Autoencoder and Variational Autoencoder

In this project I try to detect the positive cases of an imbalanced dataset in two ways. First with ordinary classifiers, and then with an Autoencoder (AE) and a Variational Autoencoder (VAE) that are trained only on normal rows and use the reconstruction error as the anomaly score.

## Note about the first version

This is a re-implementation of an earlier project of mine. In the first version I trained each model only once and the VAE looked better than the AE. In this version I trained both models with 6 seeds and I saw that this difference was only because of the seed. With more than one seed the AE and the VAE are not different on this dataset.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mishra5001/credit-card) from Kaggle. I only use `application_data.csv`.

* 307,511 rows and 122 columns
* each row is a loan application
* `TARGET = 1` is the positive class (24,825 rows, 8.07 percent)

The dataset is not in this repository because it is too large for GitHub. Download it from Kaggle: https://www.kaggle.com/datasets/mishra5001/credit-card and put `application_data.csv` next to the notebook.

## Steps

* dropping the id and the columns with more than 40 percent null
* filling the other nulls with median and mode of the train set
* one-hot encoding of the categorical columns (183 features at the end)
* 70 / 10 / 20 split for train, validation and test (stratified)
* Logistic Regression and Random Forest as first models
* PCA, AE and VAE trained on normal rows only, reconstruction error as score
* threshold is the 99th percentile of the error of normal validation rows
* AE and VAE are trained with 6 different seeds

## Results

First models on the test set (4,965 positive rows):

| Model | Accuracy | Recall | Positive rows found |
| --- | --- | --- | --- |
| Logistic Regression | 0.9195 | 0.0125 | 62 |
| Random Forest | 0.9193 | 0.0004 | 2 |
| Always normal | 0.9193 | 0.0000 | 0 |

Reconstruction error on the test set (mean and std over 6 seeds):

| Model | PR AUC | ROC AUC | Recall |
| --- | --- | --- | --- |
| PCA | 0.0802 | 0.5085 | 0.0093 |
| AE | 0.0817 ± 0.0004 | 0.5156 ± 0.0022 | 0.0088 ± 0.0009 |
| VAE (kl_weight = 0.001) | 0.0819 ± 0.0004 | 0.5165 ± 0.0016 | 0.0082 ± 0.0006 |

A random score has PR AUC about 0.0807 on this test set, because 8.07 percent of the rows are positive.

## What I found

* The classifiers have 92 percent accuracy but they find almost none of the positive rows. Random Forest is the same as a model that always says normal.
* AE and VAE are not different from each other. The difference between them is smaller than the difference between two seeds of the same model. With one seed only, either of them can look better (in seed 0 the VAE is ahead, in seed 1 the AE).
* Both of them are very close to a random score. The positive rows of this dataset do not look different from the normal rows, so their reconstruction error is not larger.
* With the default `kl_weight = 1.0` the VAE gave the same PR AUC (0.0790) for every seed. The KL term was too strong and the model ignored its input. With `kl_weight = 0.001` this problem was solved.
* Training on GPU is not exactly repeatable. The same seed gave slightly different numbers in two runs.

## Requirements

* Python 3
* TensorFlow
* scikit-learn
* pandas, NumPy, Matplotlib

## References

* Kingma and Welling, Auto-Encoding Variational Bayes
