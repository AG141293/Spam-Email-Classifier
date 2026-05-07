# 📧 Spam Email Classifier using NLP

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Colab](https://img.shields.io/badge/Open%20in-Google%20Colab-F9AB00?logo=googlecolab)

> A complete end-to-end **NLP-based Spam Email/SMS Classifier** built with Python, trained using multiple machine learning models, and evaluated with comprehensive metrics — fully compatible with **Google Colab**.

---

## 🚀 Features

- ✅ Full NLP preprocessing pipeline (tokenization, stemming, stopword removal)
- ✅ TF-IDF vectorization with unigrams + bigrams
- ✅ 5 ML models trained and compared:
  - Naive Bayes (Multinomial)
  - Logistic Regression
  - Linear SVM
  - Random Forest
  - Gradient Boosting
- ✅ Confusion Matrix, ROC Curve, Precision-Recall analysis
- ✅ Word Cloud visualization for spam vs ham
- ✅ Top predictive feature analysis
- ✅ 5-Fold Cross-Validation
- ✅ Model saved with `pickle` for reuse
- ✅ Interactive prediction function

---

## 📁 Project Structure

```
spam-email-classifier/
│
├── spam_email_classifier.py     # Main Python script (Colab ready)
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
├── spam_classifier_model.pkl    # Saved best model (auto-generated)
│
└── outputs/
    ├── eda_plots.png            # EDA visualizations
    ├── wordclouds.png           # Word cloud spam vs ham
    ├── model_comparison.png     # All models compared
    ├── best_model_results.png   # Confusion matrix + ROC curve
    └── feature_importance.png   # Top spam/ham words
```

---

## 🧠 Dataset

Uses the **UCI SMS Spam Collection Dataset** — 5,574 tagged SMS messages (ham/spam).  
Auto-downloaded via URL — no manual setup needed.

| Label | Count | Percentage |
|-------|-------|------------|
| Ham   | 4,827 | 86.6%      |
| Spam  |   747 | 13.4%      |

---

## ⚙️ NLP Preprocessing Pipeline

```
Raw Text
   ↓  Lowercase
   ↓  Remove URLs, emails, phone numbers
   ↓  Remove punctuation & special chars
   ↓  Tokenization (NLTK word_tokenize)
   ↓  Stopword Removal
   ↓  Porter Stemming
Cleaned Text → TF-IDF Features → ML Model
```

---

## 📊 Model Results (Sample)

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Naive Bayes         | 97.85%   | 0.9912  |
| Logistic Regression | 98.47%   | 0.9978  |
| Linear SVM          | 98.74%   | 0.9981  |
| Random Forest       | 97.49%   | 0.9953  |
| Gradient Boosting   | 97.31%   | 0.9944  |

> Results may vary slightly based on train/test split randomness.

---

## 🖥️ How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy and paste the cells from `spam_email_classifier.py`
4. Run all cells sequentially (`Runtime → Run All`)

Or upload the `.py` file and run:
```python
exec(open('spam_email_classifier.py').read())
```

---

## 💻 Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/spam-email-classifier.git
cd spam-email-classifier

# Install dependencies
pip install -r requirements.txt

# Run the classifier
python spam_email_classifier.py
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
nltk>=3.8.0
matplotlib>=3.6.0
seaborn>=0.12.0
wordcloud>=1.9.0
```

---

## 🔮 Interactive Prediction

```python
result = predict_spam("WINNER!! Claim your FREE prize NOW!")
print(result)
# {'label': '🚨 SPAM', 'confidence': '98.72%', ...}

result = predict_spam("Can we reschedule our meeting to Monday?")
print(result)
# {'label': '✅ HAM', 'confidence': '99.10%', ...}
```

---

## 🔭 Future Improvements

- [ ] Add BERT / transformer-based classifier
- [ ] Build a Flask/Streamlit web app
- [ ] Deploy on Hugging Face Spaces
- [ ] Add email header analysis
- [ ] Train on multilingual spam datasets

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

- [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
