# **Quora Duplicate Question Pair Identification**

## **Project Overview**

This project focuses on developing models to identify duplicate question pairs on Quora using advanced Natural Language Processing (NLP) techniques and machine learning algorithms. The primary goal is to improve the user experience by reducing redundancy in question submissions, ensuring quicker access to information, and alleviating the load on contributors.

We explore several models, including Logistic Regression, XGBoost, Siamese LSTM, and Siamese BERT Networks, to determine the most effective approach for detecting semantic similarities between question pairs. The Siamese BERT model achieved the highest performance metrics, indicating its superior ability to understand and compare question pairs.

## **Directory Structure**

- `data/`: Contains the Quora question pairs dataset (train.csv.zip).
- `models/`: Directory where the trained model weights will be saved.
- `notebooks/`: Jupyter notebooks for data exploration, model training, and evaluation.
- `README.md`: Project overview, instructions, and dependencies (this file).

## **Dependencies**

To run the code in this project, ensure that the following libraries are installed:

- Python 3.8 or later
- NumPy
- Pandas
- Matplotlib
- Seaborn
- NLTK
- Gensim
- Scikit-learn
- TensorFlow 2.x
- Transformers
- WordCloud
- XGBoost

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn nltk gensim scikit-learn tensorflow transformers wordcloud xgboost
```

## **Setup Instructions**

### **1. Clone the Repository**

Clone the project repository from GitHub:

```bash
git clone https://github.com/yourusername/quora-duplicate-question-pairs.git
cd quora-duplicate-question-pairs
```

### **2. Download the Dataset**

The Quora question pairs dataset is available on Kaggle. Download it and place the `train.csv.zip` file in the `data/` directory.

Kaggle Dataset URL: [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs/data)

### **3. Run the Jupyter Notebooks**

All the steps for data preprocessing, model training, and evaluation are provided within the Jupyter notebooks. The main notebook to run is:

- `notebooks/quora_duplicate_question_pairs.ipynb`

This notebook includes the following steps:

1. **Data Preprocessing**: Clean and prepare the dataset for modeling, including text preprocessing and handling missing values.
2. **Exploratory Data Analysis (EDA)**: Visualize the data to understand class distribution, question lengths, and word frequencies.
3. **Model Training**: Train different models (Logistic Regression, XGBoost, Siamese LSTM, Siamese BERT) on the processed data.
4. **Model Evaluation**: Evaluate the performance of each model using metrics like accuracy, precision, recall, F1-score, and ROC AUC.

### **4. Save Model Weights**

During training, the model weights will be automatically saved in the `models/` directory. You can reload these weights later for further evaluation or inference.

### **5. Visualize Results**

The notebook includes cells that generate visualizations such as confusion matrices and ROC curves for each model. These results help in comparing the models and understanding their performance.

## **Project Notes**

- The Siamese BERT model requires significant computational resources. It is recommended to use a TPU or GPU for training.
- Early stopping and learning rate reduction callbacks are used in the training process to prevent overfitting.
- Ensure your system has sufficient memory and processing power, especially when training large models like BERT.

## **Future Work**

- Fine-tune the BERT model further for improved accuracy.
- Explore other transformer models such as GPT-3 or RoBERTa.
- Implement data augmentation techniques to enhance model generalization.
- Investigate unsupervised methods for duplicate question detection.

## **Contributors**

- Kahan Dhaneshbhai Sheth (sheth.kah@northeastern.edu)
