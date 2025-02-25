# Predicting Vehicle Drag Coefficient Using Ensemble Machine Learning

This repository contains a project that leverages machine learning techniques to predict the drag coefficient (Cd) of vehicles based on features extracted from CAD images. The workflow uses data preprocessing, ensemble learning (a stacking regressor combining Gradient Boosting and Neural Networks), and model evaluation using Mean Squared Error.

## Introduction

In automotive design, accurately estimating the aerodynamic drag coefficient is crucial for improving fuel efficiency and vehicle performance. This project aims to predict the drag coefficient (`Average Cd`) using a variety of features derived from CAD images. By combining classical machine learning models with deep learning approaches in an ensemble framework, the model benefits from both high interpretability and robust nonlinear approximations. The ensemble approach employs:
- **Gradient Boosting Regressor (GBR):** Capturing complex nonlinear relationships.
- **Neural Network (via Keras):** Providing flexible, deep feature extraction capabilities.

A final Gradient Boosting model is used to aggregate predictions from these base models, leading to improved overall performance.

## Features

- **Data Preprocessing:** Uses `scikit-learn`'s `ColumnTransformer` to apply `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.
- **Ensemble Modeling:** Implements a stacking regressor that combines:
  - A **Gradient Boosting Regressor** as a classical machine learning model.
  - A **Keras-based Neural Network** as a deep learning model.
- **Model Evaluation:** Measures performance using Mean Squared Error (MSE).

## Future Work
Hyperparameter Tuning: Experiment with different network architectures and gradient boosting parameters.
Data Augmentation: Explore additional features extracted from CAD images.
Cross-validation: Incorporate more robust cross-validation strategies to ensure model generalization.

## Contributors
Koushik Balaji P , Roshaun Infant R , Shree Pranav S, Joel Ebenezer P

## Contributing
Contributions to this project are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

License
This project is licensed under the MIT License.
