# News Category Classification

This repository contains code and results for a multiclass classification project on a news category dataset. The goal is to classify news headlines and short descriptions into predefined categories.

## Repository Structure

```data/
├── processed/
│   ├── test.csv
│   └── train.csv
└── raw/
└── News_Category_Dataset_v3.json
logs/
├── basic_algos.txt
├── boostings_alogs.txt
├── cnn.txt
└── results.md
notebooks/
├── 01_data_ingestion.ipynb
├── 02_classical_baselines.ipynb
├── 03_boosting.ipynb
└── 04_cnn_text.ipynb
requirements.txt
README.md
```


- `data/`: Contains the raw and processed datasets.
    - `raw/`: Stores the original JSON dataset.
    - `processed/`: Stores the cleaned and split CSV files (`train.csv`, `test.csv`).
- `logs/`: Stores text files containing the output and results of different model runs.
    - `results.md`: Contains the key learnings and overview of the differnet models' outputs. 
- `notebooks/`: Contains the Jupyter notebooks detailing each step of the project.
    - `01_data_ingestion.ipynb`: Notebook for loading, cleaning, and splitting the data.
    - `02_classical_baselines.ipynb`: Notebook for training and evaluating classical machine learning models.
    - `03_boosting.ipynb`: Notebook for training and evaluating boosting algorithms.
    - `04_cnn_text.ipynb`: Notebook for building and training a 1D Convolutional Neural Network for text classification.
- `requirements.txt`: Lists the Python packages required to run the code.
- `README.md`: This file, providing an overview of the project.

## Project Overview

This project explores various machine learning approaches for classifying news articles into different categories. The process involves:

1.  **Data Ingestion and Preprocessing:** Loading the raw JSON data, cleaning the text (lowercasing, removing URLs and punctuation), and splitting the data into training and testing sets while preserving the class distribution (stratified split).
2.  **Classical Machine Learning Baselines:** Training and evaluating several traditional machine learning models using TF-IDF for feature extraction:
    -   Logistic Regression (with Softmax)
    -   Decision Tree
    -   Random Forest
    -   K-Nearest Neighbors (KNN)
    -   Naive Bayes
3.  **Boosting Algorithms:** Training and evaluating gradient boosting models:
    -   XGBoost
    -   LightGBM
    -   CatBoost
4.  **Deep Learning Model:** Building and training a 1D Convolutional Neural Network (CNN) to learn hierarchical features from the text data.

## Notebook Details

### 1. `01_data_ingestion.ipynb`

This notebook handles the initial steps of the project:

-   Loads the raw news category dataset from `data/raw/News_Category_Dataset_v3.json`.
-   Extracts the `headline`, `category`, and `short_description` columns.
-   Cleans the text data by:
    -   Converting to lowercase.
    -   Removing URLs.
    -   Removing punctuation.
    -   Removing extra whitespace.
-   Performs a stratified train-test split (80% train, 20% test) to maintain the proportion of categories in both sets.
-   Saves the processed training and testing data as `train.csv` and `test.csv` in the `data/processed/` directory.

### 2. `02_classical_baselines.ipynb`

This notebook focuses on training and evaluating classical machine learning models:

-   Loads the processed training and testing data.
-   Combines the `headline` and `short_description` for text representation.
-   Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text data into numerical feature vectors.
-   Trains and evaluates the following models:
    -   **Logistic Regression:** Achieved an accuracy of 0.5942 and a macro F1 score of 0.4456. Performed well on categories like 'DIVORCE', 'HOME & LIVING', and 'POLITICS' but struggled with others.
    -   **Decision Tree:** Achieved an accuracy of 0.4189 and a macro F1 score of 0.2877. Showed signs of overfitting and poor generalization on text data.
    -   **Random Forest:** Achieved an accuracy of 0.5008 and a macro F1 score of 0.3141. Performed better than a single decision tree but still below Logistic Regression.
    -   **K-Nearest Neighbors:** Achieved a low accuracy of 0.1761 and a macro F1 score of 0.1562, indicating it's not well-suited for high-dimensional sparse text data.
    -   **Naive Bayes:** Achieved an accuracy of 0.5294 and a macro F1 score of 0.2966. Showed decent accuracy but imbalanced performance across categories.
-   Prints the accuracy, macro F1 score, and classification report for each model.

### 3. `03_boosting.ipynb`

This notebook explores gradient boosting algorithms:

-   Reuses the TF-IDF features generated in the previous notebook.
-   Trains and evaluates the following boosting models:
    -   **XGBoost:** Training was noted to be slow and memory-intensive for this dataset.
    -   **LightGBM:** Achieved an accuracy of 0.5654 and a macro F1 score of 0.4434, offering a good balance of performance and training speed (around 3-4 minutes).
    -   **CatBoost:** Training time was also recorded.
-   Includes considerations for early stopping (though the implementation details might be in the notebook).
-   Mentions plotting feature importances (implementation might be in the notebook).

### 4. `04_cnn_text.ipynb`

This notebook implements a 1D Convolutional Neural Network for text classification:

-   Loads the processed data.
-   Tokenizes the text data using `tensorflow.keras.preprocessing.text.Tokenizer`.
-   Converts text sequences to integer sequences and pads them to a uniform length.
-   Converts categorical labels using `tensorflow.keras.utils.to_categorical`.
-   Builds a sequential 1D CNN model with the following layers:
    -   Embedding layer.
    -   Conv1D layer with ReLU activation.
    -   GlobalMaxPooling1D layer.
    -   Dense layer with ReLU activation and Dropout.
    -   Output Dense layer with Softmax activation.
-   Compiles the model with the Adam optimizer, categorical cross-entropy loss, and accuracy metric.
-   Trains the model for a specified number of epochs and batch size, with validation on the test set.
-   Evaluates the model on the test set and prints the test accuracy and F1 score.
-   Generates and displays plots of training and validation accuracy and loss over epochs.
-   Prints the classification report.

## Results Summary

The project explored several machine learning models for news category classification. Key findings include:

-   Classical models like Logistic Regression provided a strong baseline.
-   Tree-based models (Decision Tree and Random Forest) underperformed compared to Logistic Regression, likely due to the high dimensionality of the TF-IDF features.
-   KNN was not suitable for this type of sparse text data.
-   Boosting algorithms like LightGBM offered competitive performance with reasonable training times.
-   The 1D CNN achieved comparable accuracy to the best classical models and a better overall F1 score, demonstrating the potential of deep learning for this task.

## Next Steps

The next phase of this project will involve exploring more advanced techniques, specifically:

-   **Transformers and Large Language Models (LLMs):** Investigating the use of pre-trained transformer models and LLMs for this classification task.
-   **Fine-tuning:** Experimenting with different fine-tuning strategies to adapt these powerful models to the specific news category dataset.

Stay tuned for further developments in this repository!

## Requirements

To run the code in this repository, you will need the following Python packages installed. You can install them using pip:

```bash
pip install -r requirements.txt