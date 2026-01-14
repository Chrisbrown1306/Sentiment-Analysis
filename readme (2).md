# Sentiment Analysis of Text Data using Machine Learning

A comprehensive text classification project implementing Natural Language Processing (NLP) techniques with machine learning models for sentiment analysis.

## 🎯 Features

- Text Preprocessing: Tokenization, stop-word removal, stemming
- Feature Extraction: TF-IDF vectorization
- Machine Learning Models: Naive Bayes and Support Vector Machine (SVM)
- Performance Evaluation: Confusion matrix, accuracy metrics, classification reports
- Visualization: Sentiment distribution charts and model comparison plots

## 🛠️ Technologies Used

- Python 3.8+
- Scikit-learn: Machine learning algorithms
- NLTK: Natural language processing
- Pandas: Data manipulation
- NumPy: Numerical computations
- Matplotlib & Seaborn: Data visualization

## 📋 Prerequisites

```bash
Python 3.8 or higher
pip (Python package installer)
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis-ml.git
cd sentiment-analysis-ml
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Basic Usage

Run the main script:
```bash
python sentiment_analysis.py
```

### Using Custom Dataset

If you have your own dataset (CSV file with 'text' and 'sentiment' columns):

```python
from sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
df = analyzer.load_data('your_dataset.csv')
```

### Making Predictions

```python
from sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# After training the model
text = "This product is amazing!"
prediction = analyzer.predict_sentiment(text, model='nb')  # or 'svm'
print(f"Sentiment: {prediction}")
```

## 📁 Project Structure

```
sentiment-analysis-ml/
│
├── sentiment_analysis.py      # Main script with complete implementation
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # License file
├── .gitignore                  # Git ignore file
│
├── data/                       # Dataset folder (optional)
│   └── your_dataset.csv
│
└── output/                     # Generated visualizations
    ├── sentiment_distribution.png
    ├── confusion_matrix_naive_bayes.png
    ├── confusion_matrix_svm.png
    └── model_comparison.png
```

## 🔍 Methodology

### 1. Text Preprocessing
- Convert text to lowercase
- Remove URLs, mentions, and hashtags
- Remove punctuation and numbers
- Tokenization using NLTK
- Stop-word removal
- Stemming using Porter Stemmer

### 2. Feature Extraction
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Maximum features: 5000

### 3. Model Training
- **Naive Bayes**: Multinomial Naive Bayes classifier
- **SVM**: Support Vector Machine with linear kernel

### 4. Evaluation Metrics
- Accuracy Score
- Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report

## 📈 Results

The project generates several visualizations:

1. **Sentiment Distribution**: Bar and pie charts showing class distribution
2. **Confusion Matrices**: Heatmaps for both models
3. **Model Comparison**: Bar chart comparing model accuracies

### Sample Output

```
Naive Bayes Model Evaluation
Accuracy: 0.9250

SVM Model Evaluation
Accuracy: 0.9350
```

## 🔧 Customization

### Modify Preprocessing

Edit the `preprocess_text` method in the `SentimentAnalyzer` class:

```python
def preprocess_text(self, text):
    # Add your custom preprocessing steps
    pass
```

### Change Model Parameters

```python
self.nb_model = MultinomialNB(alpha=1.0)
self.svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
```

### Adjust TF-IDF Settings

```python
self.vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)
```

## 📝 Dataset Format

Your CSV file should have the following structure:

```csv
text,sentiment
"I love this product!",positive
"Terrible experience",negative
"It's okay",neutral
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE]() file for details.

## 👤 Author

Om Naik
- GitHub: [@ChrisBrown1306](https://github.com/ChrisBrown1306)


## 🙏 Acknowledgments

- NLTK team for natural language processing tools
- Scikit-learn developers for machine learning algorithms
- Open-source community for inspiration and support

## 📧 Contact

For questions or feedback, please reach out:
- Email: omnaik6969@gmail.com
- Project Link: 

---

⭐ If you found this project helpful, please give it a star!
