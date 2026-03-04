# StyleSense: Fashion Forward Forecasting
## Project Overview

StyleSense is a rapidly growing online women's fashion retailer known for its trendy and affordable clothing. As the company expanded, a large number of product reviews began to accumulate. While customers frequently provide detailed written feedback, many reviews do not include an explicit recommendation indicator.

The objective of this project is to build a machine learning pipeline capable of predicting whether a customer would recommend a product based on available information such as:

Customer demographics

Product category attributes

Review text content

Engagement metrics (e.g., positive feedback count)

By automating this prediction process, StyleSense can gain deeper insights into customer satisfaction, identify trends in product perception, and enhance the overall shopping experience.

The final solution integrates data preprocessing, natural language processing (NLP), and classification modeling within a unified pipeline, ensuring reproducibility and scalability.

## Dataset Summary

The dataset contains 18,442 customer reviews collected from an e-commerce platform for women's clothing products.

Features           Feature Description

Clothing ID:	Identifier for the clothing product
Age:	Age of the reviewer
Title:	Title of the review
Review Text:	Full text of the customer review
Positive Feedback Count:	Number of customers who found the review helpful
Division Name:	High-level product division
Department Name:	Product department
Class Name:	Product class
Target Variable         Target	Description

Recommended IND:	Binary variable indicating whether the reviewer recommends the product (1 = recommended, 0 = not recommended)

The dataset contains both structured data (numerical and categorical) and unstructured text data, making it well suited for a machine learning pipeline that integrates NLP techniques.

## Key Findings & Challenges
Key Insights from Data Exploration

-The dataset is class imbalanced, with approximately 81.6% of reviews recommending the product.

-Most reviewers fall within the 30–50 age range, suggesting that the primary customer segment consists of middle-aged shoppers.

-Review length typically ranges between 100 and 450 characters, with a visible spike near the maximum character limit.

-The Positive Feedback Count distribution is highly skewed, where most reviews receive very few interactions.

Challenges

-Class imbalance, which may bias the model toward predicting recommended products.

-High-dimensional text data generated from TF-IDF vectorization.

-Integrating numeric, categorical, and text features into a single machine learning pipeline.

-Addressing these challenges required careful preprocessing and model tuning.

## Model Pipeline Architecture

To ensure consistency and reproducibility, the project uses a scikit-learn pipeline architecture that integrates all preprocessing and modeling steps.

The pipeline includes:

1. Numeric Feature Processing

    -Missing value imputation (median)

    -Feature scaling using StandardScaler

2. Categorical Feature Processing

    -Missing value imputation

    -One-Hot Encoding for categorical variables

3. Text Processing (NLP)

    -Text normalization

    -Stop-word removal

    -Feature extraction using TF-IDF Vectorization

4. Classification Model

    -The pipeline uses Logistic Regression as the classification model.

    -This architecture ensures that the same preprocessing steps are applied during both training and inference, preventing data leakage.

Performance & Evaluation

Model performance was evaluated using a held-out test set (10% of the dataset).

Baseline Model Performance

Accuracy: 87.6%

Key observations:

    -The model performs very well in identifying recommended products.

    -However, recall for the non-recommended class was relatively low due to class imbalance.

Tuned Model Performance

After applying GridSearchCV for hyperparameter tuning, performance improved:

Accuracy: 88.8%

Improvements observed:

    -Better detection of non-recommended reviews

    -More balanced performance between the two classes

    -Reduced bias toward the majority class

Evaluation metrics used:

    -Accuracy

    -Precision

    -Recall

    -F1-score

    -Confusion Matrix

These metrics provide a more complete understanding of model performance, especially under class imbalance.

## Files in Repository
File	                                Description
StyleSense_Fashion_Forecasting.ipynb	Main notebook containing data exploration, pipeline creation, model training, and evaluation
data/reviews.csv	                    Dataset containing customer reviews
README.md	                            Project documentation