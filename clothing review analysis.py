# Women's Clothing E-commerce Review Analysis
# Includes: EDA, Sentiment Analysis, Text Mining, Modeling
# -------------------------------------------------------

# ========== Package Imports ==========
import os
import gc
import re
import string
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm, skew, mode

from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from keras.metrics import Precision, Recall

# ========== Load Dataset ==========
data = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
df = data.copy()

# ========== Preprocessing ==========
df.drop(columns=["Unnamed: 0"], inplace=True)
df.dropna(subset=["Review Text"], inplace=True)
df.reset_index(drop=True, inplace=True)

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if col == 'Title':
            df[col] = df[col].fillna(' ')
        else:
            df[col] = df[col].fillna('Blank')

df.drop_duplicates(inplace=True)

# ========== EDA ==========
# Histograms of continuous variables by recommendation
cont_cols = ['Clothing ID', 'Age', 'Positive Feedback Count']
fig, ax = plt.subplots(1, 3, figsize=(20, 4))
for idx, feature in enumerate(cont_cols):
    sns.histplot(data=df, x=feature, hue='Recommended IND', bins=20, ax=ax[idx], palette=['#80CBC4', '#4F8A8B'], multiple='stack')
    ax[idx].set_title(f'{feature} by Recommendation')
plt.tight_layout()
plt.show()

# Ratings vs Recommendation Rate
recommendation_rates = df.groupby("Rating")["Recommended IND"].mean().reset_index()
plt.figure(figsize=(8,5))
sns.barplot(x="Rating", y="Recommended IND", data=recommendation_rates, palette="Set2")
plt.title("Rating vs. Recommendation Rate")
plt.ylim(0, 1)
plt.show()

# ========== Text Preprocessing ==========
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

df["Cleaned Review"] = df["Review Text"].apply(clean_text)

# ========== Word Clouds ==========
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["Cleaned Review"]))
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of All Reviews")
plt.show()

# ========== Sentiment Labeling ==========
def assign_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["Rating"].apply(assign_sentiment)

# ========== Sentiment Visualization ==========
sentiment_counts = df["Sentiment"].value_counts(normalize=True) * 100
sentiment_df = df["Sentiment"].value_counts().reset_index()
sentiment_df.columns = ["Sentiment", "Count"]
sentiment_df["Percentage"] = sentiment_df["Sentiment"].map(lambda x: sentiment_counts[x])

plt.figure(figsize=(8, 5))
sns.barplot(x="Sentiment", y="Count", data=sentiment_df, palette="Set2", order=["Positive", "Neutral", "Negative"])
plt.title("Sentiment Distribution")
plt.show()

# ========== Model Training (Sentiment Classification) ==========
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Cleaned Review"])
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": MultinomialNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

# Plot accuracy comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=[res["accuracy"] for res in results.values()], palette="Set2")
plt.title("Model Accuracy Comparison")
plt.show()

# ========== Feature Engineering + Recommendation Prediction ==========
df["Polarity"] = df["Review Text"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["Subjectivity"] = df["Review Text"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df["Word Count"] = df["Review Text"].apply(lambda x: len(x.split()))
df["Avg Word Length"] = df["Review Text"].apply(lambda x: np.mean([len(w) for w in x.split()]))

X_text = vectorizer.fit_transform(df["Cleaned Review"])
X_num = df[["Rating", "Polarity", "Subjectivity", "Word Count", "Avg Word Length"]].values
X_combined = np.hstack((X_text.toarray(), X_num))
y_binary = df["Recommended IND"]

X_train, X_test, y_train, y_test = train_test_split(X_combined, y_binary, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Recommendation Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ========== Feature Importance ==========
feature_names = list(vectorizer.get_feature_names_out()) + ["Rating", "Polarity", "Subjectivity", "Word Count", "Avg Word Length"]
importances = rf_model.feature_importances_
top_features = sorted(zip(importances, feature_names), reverse=True)[:20]

plt.figure(figsize=(10, 5))
sns.barplot(x=[val[0] for val in top_features], y=[val[1] for val in top_features])
plt.title("Top 20 Important Features for Recommendation")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ========== Topic Modeling (LDA) ==========
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_text)
words = np.array(vectorizer.get_feature_names_out())

for i, topic in enumerate(lda.components_):
    top_words = [words[j] for j in topic.argsort()[:-11:-1]]
    print(f"Topic {i+1}: {', '.join(top_words)}")