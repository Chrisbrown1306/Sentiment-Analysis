"""
Sentiment Analysis of Text Data using Machine Learning
Author: Om Naik
Description: Text classification using NLP techniques with Naive Bayes and SVM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

class SentimentAnalyzer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.nb_model = MultinomialNB()
        self.svm_model = SVC(kernel='linear', random_state=42)
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'\@\w+|\#', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply stemming
        tokens = [self.stemmer.stem(word) for word in tokens 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def load_data(self, filepath=None):
        """Load and prepare dataset"""
        if filepath:
            df = pd.read_csv(filepath)
        else:
            # Create sample dataset for demonstration
            data = {
                'text': [
                    "I love this product! It's amazing and works perfectly.",
                    "Terrible experience. Would not recommend to anyone.",
                    "Pretty good overall, met my expectations.",
                    "Waste of money. Very disappointed with the quality.",
                    "Absolutely fantastic! Best purchase I've made.",
                    "Not bad, but could be better. Average product.",
                    "Horrible customer service and poor quality.",
                    "I'm very satisfied with this purchase. Great value!",
                    "Decent product but overpriced for what you get.",
                    "Outstanding quality and fast shipping. Highly recommend!",
                    "The worst product I've ever bought. Total disaster.",
                    "It's okay. Nothing special but does the job.",
                    "Excellent! Exceeded all my expectations.",
                    "Poor quality. Broke after just one use.",
                    "Really happy with this. Worth every penny!",
                    "Mediocre at best. Not worth the price.",
                    "Fantastic product! Will definitely buy again.",
                    "Complete waste of time and money.",
                    "Good quality and reasonable price. Satisfied.",
                    "Awful. Do not buy this under any circumstances."
                ] * 50,  # Repeat to create larger dataset
                'sentiment': ['positive', 'negative', 'neutral', 'negative', 'positive',
                            'neutral', 'negative', 'positive', 'neutral', 'positive',
                            'negative', 'neutral', 'positive', 'negative', 'positive',
                            'neutral', 'positive', 'negative', 'positive', 'negative'] * 50
            }
            df = pd.DataFrame(data)
        
        return df
    
    def prepare_data(self, df):
        """Preprocess and split data"""
        print("Preprocessing text data...")
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        X = df['cleaned_text']
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF Vectorization
        print("Applying TF-IDF vectorization...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train Naive Bayes and SVM models"""
        print("\nTraining Naive Bayes model...")
        self.nb_model.fit(X_train, y_train)
        
        print("Training SVM model...")
        self.svm_model.fit(X_train, y_train)
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        print(f"\n{'='*60}")
        print(f"{model_name} Model Evaluation")
        print(f"{'='*60}")
        
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return y_pred, cm, accuracy
    
    def plot_confusion_matrix(self, cm, model_name, labels):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300)
        plt.show()
    
    def plot_sentiment_distribution(self, df):
        """Visualize sentiment distribution"""
        plt.figure(figsize=(10, 6))
        
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        plt.subplot(1, 2, 1)
        sentiment_counts.plot(kind='bar', color=colors)
        plt.title('Sentiment Distribution (Count)')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors)
        plt.title('Sentiment Distribution (Percentage)')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png', dpi=300)
        plt.show()
    
    def plot_model_comparison(self, nb_acc, svm_acc):
        """Compare model accuracies"""
        plt.figure(figsize=(8, 6))
        models = ['Naive Bayes', 'SVM']
        accuracies = [nb_acc, svm_acc]
        colors = ['#3498db', '#e67e22']
        
        bars = plt.bar(models, accuracies, color=colors)
        plt.title('Model Performance Comparison')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300)
        plt.show()
    
    def predict_sentiment(self, text, model='nb'):
        """Predict sentiment of new text"""
        cleaned = self.preprocess_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        
        if model == 'nb':
            prediction = self.nb_model.predict(vectorized)[0]
        else:
            prediction = self.svm_model.predict(vectorized)[0]
        
        return prediction

def main():
    """Main execution function"""
    print("="*60)
    print("Sentiment Analysis using Machine Learning")
    print("="*60)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Load data
    print("\nLoading dataset...")
    df = analyzer.load_data()
    print(f"Dataset shape: {df.shape}")
    
    # Plot sentiment distribution
    analyzer.plot_sentiment_distribution(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test = analyzer.prepare_data(df)
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Train models
    analyzer.train_models(X_train, y_train)
    
    # Evaluate Naive Bayes
    nb_pred, nb_cm, nb_acc = analyzer.evaluate_model(
        analyzer.nb_model, X_test, y_test, "Naive Bayes"
    )
    
    # Evaluate SVM
    svm_pred, svm_cm, svm_acc = analyzer.evaluate_model(
        analyzer.svm_model, X_test, y_test, "SVM"
    )
    
    # Plot confusion matrices
    labels = sorted(df['sentiment'].unique())
    analyzer.plot_confusion_matrix(nb_cm, "Naive Bayes", labels)
    analyzer.plot_confusion_matrix(svm_cm, "SVM", labels)
    
    # Compare models
    analyzer.plot_model_comparison(nb_acc, svm_acc)
    
    # Test predictions
    print("\n" + "="*60)
    print("Testing Model Predictions")
    print("="*60)
    
    test_samples = [
        "This is absolutely wonderful! I love it!",
        "Terrible product. Very disappointed.",
        "It's okay, nothing special."
    ]
    
    for sample in test_samples:
        nb_result = analyzer.predict_sentiment(sample, 'nb')
        svm_result = analyzer.predict_sentiment(sample, 'svm')
        print(f"\nText: {sample}")
        print(f"Naive Bayes Prediction: {nb_result}")
        print(f"SVM Prediction: {svm_result}")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()