# Results Summary: News Category Multiclass Classification

This document summarizes the performance of various machine learning models applied to the news category classification task. The goal was to classify news headlines and short descriptions into predefined categories. We experimented with classical machine learning models using TF-IDF features, gradient boosting algorithms, and a 1D Convolutional Neural Network (CNN).

## Classical Machine Learning Models

The following table summarizes the performance of the classical machine learning models:

| Model                 | Accuracy | Macro F1 |
|-----------------------|----------|----------|
| Logistic Regression   | 0.5942   | 0.4456   |
| Decision Tree         | 0.4189   | 0.2877   |
| Random Forest         | 0.5008   | 0.3141   |
| k-Nearest Neighbors   | 0.1761   | 0.1562   |
| Naive Bayes           | 0.5294   | 0.2966   |

**Key Learnings:**

* **Logistic Regression:** Achieved the highest accuracy among the classical models. It demonstrated strong performance on certain categories ('DIVORCE', 'HOME & LIVING', 'POLITICS') but struggled with others ('ARTS', 'ARTS & CULTURE'), suggesting potential class imbalance issues or difficulty in capturing nuanced semantic differences for some categories using TF-IDF alone.
* **Decision Tree:** Performed significantly worse than other models, indicating that a single decision tree is prone to overfitting on high-dimensional text data and struggles to generalize effectively.
* **Random Forest:** Showed improvement over a single decision tree due to the ensemble nature, but still lagged behind Logistic Regression. The complexity of the text data and potential for many irrelevant features in TF-IDF might limit its effectiveness.
* **k-Nearest Neighbors:** Exhibited very poor performance. This distance-based algorithm is likely not well-suited for the high-dimensional and sparse feature space created by TF-IDF. The notion of 'nearest neighbors' becomes less reliable in such spaces.
* **Naive Bayes:** Provided a reasonable accuracy but a lower macro F1 score compared to Logistic Regression. This suggests that while it performs adequately on some majority classes, its performance is less balanced across all categories. The independence assumption of Naive Bayes might not hold well for text data.

## Boosting Algorithms

The following table summarizes the performance of the boosting algorithm:

| Model       | Accuracy | Macro F1 | Training Time (approx.) |
|-------------|----------|----------|-------------------------|
| LightGBM    | 0.5654   | 0.4434   | ~3-4 minutes            |
| XGBoost     | (Results not fully captured due to long training time) | (Results not fully captured due to long training time) | >20 minutes              |
| CatBoost    | (Results not fully captured due to long training time) | (Results not fully captured due to long training time) | >20 minutes              |

**Key Learnings:**

* **LightGBM:** Demonstrated competitive performance, achieving a macro F1 score close to Logistic Regression with a reasonable training time. Its efficiency makes it a good candidate for further tuning.
* **XGBoost and CatBoost:** While potentially powerful, these algorithms were significantly slower to train on this dataset, making rapid experimentation and hyperparameter tuning challenging within the initial scope.

## 1D Convolutional Neural Network (CNN)

| Metric        | Value    |
|---------------|----------|
| Accuracy      | 0.5909   |
| Weighted F1   | 0.5756   |
| Macro F1      | 0.43     |

**Key Learnings:**

* The 1D CNN achieved an accuracy comparable to the best classical model (Logistic Regression) and a better weighted F1 score. This indicates that the CNN is capable of learning meaningful hierarchical features from the word embeddings, capturing contextual information that might be missed by TF-IDF.
* The macro F1 score is slightly lower than the weighted F1, suggesting some imbalance in performance across different categories, which is consistent with the observations from the classical models.

## Overall Learnings and Next Steps

* **Simpler models can be effective baselines:** Logistic Regression with TF-IDF provided a strong starting point.
* **Tree-based models might require different feature representations:** The sparse TF-IDF vectors might not be optimal for decision tree-based models. Exploring other feature engineering techniques or different model parameters could be beneficial.
* **KNN is not well-suited for high-dimensional sparse text data.**
* **Boosting algorithms offer potential but can be computationally expensive:** LightGBM showed promise in terms of performance and efficiency.
* **Deep learning models like 1D CNNs can capture complex textual patterns:** The CNN's performance indicates the value of learning embeddings and local contextual features directly from the text.
* **Class imbalance is likely an issue:** The varying performance across different categories in all models suggests that some categories might have significantly different numbers of samples, impacting the training process.

**Next Steps:**

The next phase of this project will focus on leveraging more advanced techniques:

* **Transformers and Large Language Models (LLMs):** Exploring pre-trained models like BERT, RoBERTa, etc., and fine-tuning them for this news category classification task. These models excel at capturing semantic relationships and context in text.
* **Addressing Class Imbalance:** Implementing techniques like oversampling, undersampling, or using class-weighted loss functions to improve the performance on underrepresented categories.
* **Hyperparameter Tuning:** Performing more extensive hyperparameter tuning for the promising models (Logistic Regression, LightGBM, CNN) to potentially improve their performance further.
* **Exploring Different Text Embeddings:** Experimenting with pre-trained word embeddings (e.g., Word2Vec, GloVe, FastText) as input to the CNN and other models.