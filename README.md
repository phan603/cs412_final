# cs412_final
CS 412 Final Project

This project analyzes and models the Yelp Academic Dataset using various data mining and NLP techniques. The analysis is conducted through Jupyter Notebooks using Python, with models including Logistic Regression, SVM, VADER, fine-tuned BERT and other machine learning pipelines.

## 📁 Dataset

To reproduce the results, download the dataset from Kaggle:

👉 [Yelp Dataset on Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data?select=yelp_academic_dataset_business.json)

Make sure to download only the `yelp_academic_dataset_business.json` file and place them in a `data/` directory at the root of this project.

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/yelp-nlp-analysis.git
   cd yelp-nlp-analysis

2. **Open Jupyter Notebook Environment**
   Visual Studio Code or Google Colab are both viable options to run our Jupyter notebooks or view the results of the code we have.

3. **Install the required packages**
  We used the following libraries in our code repository:

pandas

numpy

torch

transformers

matplotlib

seaborn

scikit-learn

tqdm

nltk

To install, you can use pip as follows:
`pip install pandas torch transformers matplotlib seaborn scikit-learn tqdm nltk`

4. **View the dataset, models and evaluation metrics we used!**

## How to Run the Code
### Cleaning Dataset:
1) Download `CS412_Project.ipynb` and place it in your project folder with the `yelp_academic_dataset_business.json` file.
2) Open `CS412_Project.ipynb`, and uncomment code in the first, second, and third cells to load the json data depending on your coding interface.
3) Run the the rest of the cells (from fourth to ninth cells) to convert the json file to a csv `yelp_reviews.csv` and perform data cleaning.

### Unigram Naive Bayes:
1) Place the `yelp_reviews.csv` in the same folder as `Unigram_Naive_Bayes.ipynb`.
2) Run all cells in `Unigram_Naive_Bayes.ipynb` in order to load the dataset, train the unigram model on the dataset, and output results.

### Bigram Naive Bayes:
1) Place the `yelp_reviews.csv` in the same folder as `Bigram_Naive_Bayes.ipynb`.
2) Run all cells in `Bigram_Naive_Bayes.ipynb` in order to load the dataset, train the unigram model on the dataset, and output results.

### Logistic Regression:
1) Place the `yelp_reviews.csv` in the same folder as `Logistic_Regression_Baseline.ipynb`.
2) Run all cells in `Logistic_Regression_Baseline.ipynb` in order to load the dataset, train the Logistic Regression model on the dataset, and output results.

### VADER:

### SVM:
1) Place the `yelp_reviews.csv` in the same folder as `CS412_Project_SVM.ipynb`.
2) Run all cells in `CS412_Project_SVM.ipynb` in order to load the dataset, train the unigram model on the dataset, and output results.

### BERT:
1) Place the `yelp_reviews.csv` in the same folder as `BERT_Classification.ipynb`.
2) Run all cells in `BERT_Classification.ipynb` in order to load the dataset, train the BERT model on the dataset, and output results.
