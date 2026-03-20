Machine Learning Analysis of Gas Properties

This project implements a complete machine learning workflow for analyzing gas properties and predicting gas quality levels. The pipeline includes data preprocessing, clustering, and supervised classification using neural networks.

The goal is to evaluate how different models perform on standardized gas property data and compare their effectiveness.


The dataset contains physical gas properties used to predict quality levels.

Features:

Temperature (T)

Pressure (P)

Critical Temperature (TC)

Specific Volume (SV)

Target:

Wobbe Index (Idx), converted into three classes:

Regular

Medium

Premium

Data Preprocessing

The dataset is standardized using z-score normalization:

z = (x - mean) / standard deviation

This ensures:

Mean approximately equal to 0

Standard deviation approximately equal to 1

All features contribute equally during training

Run preprocessing:

python data_preprocess.py
Clustering

The following clustering methods are implemented:

K-Means

Gaussian Mixture Model (EM)

Self Organizing Map

Evaluation

Clustering performance is evaluated using the Silhouette Score.

Results:

K-Means: 0.4211

GMM (EM): 0.3318

SOM: 0.3533

K-Means produced the best cluster separation.

Run clustering:

python k_means_clustering.py
python em_clustering.py
python SOM_clustering.py

Classification
Multi Layer Perceptron (MLP)

Test Accuracy: 80.70%

Macro F1 Score: 0.8066

The MLP achieved the best overall performance and captured non linear relationships effectively.

Radial Basis Function (RBF)

Number of centers: 150

Test Accuracy: 69.22%

Macro F1 Score: 0.6862

The RBF model was more sensitive to hyperparameters and required more tuning.

Model Comparison

The MLP model was easier to tune and achieved higher accuracy and F1 score. The RBF model required careful selection of the number of centers and kernel width, making it more complex.

The difference in performance is due to how each model learns patterns. MLP uses layered transformations to learn global relationships, while RBF relies on localized responses around cluster centers.

Technologies Used

Python

Pandas

NumPy

Scikit-learn

How to Run

Install dependencies:

pip install pandas numpy scikit-learn

Run full pipeline:

python data_preprocess.py
python k_means_clustering.py
python em_clustering.py
python SOM_clustering.py
python evaluation.py
python classification_data.py
python mlp_classifier.py
python rbf_classifier.py
Notes

Standardization is critical for clustering and neural networks

K-Means provided the most well separated clusters

MLP outperformed RBF in both accuracy and stability

RBF performance depends heavily on the number of centers
