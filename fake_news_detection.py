import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_excel("fake_news_dataset.xlsx")

# Ensure necessary columns are present
if not {'Content', 'Label', 'Source'}.issubset(df.columns):
    raise ValueError("Dataset must contain 'Content', 'Label', and 'Source' columns.")

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(tokens)

# Apply preprocessing
df['Processed_Content'] = df['Content'].astype(str).apply(preprocess_text)

# Convert labels to numerical values
df['Label'] = df['Label'].map({'Real': 1, 'Fake': 0})

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['Processed_Content'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['Label'], test_size=0.2, random_state=42)

# Train Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Fake News Detection - Confusion Matrix")
plt.show()

# Visualize the distribution of real vs. fake news
plt.figure(figsize=(8, 5))
sns.countplot(x=df['Label'].map({1: "Real", 0: "Fake"}), palette="coolwarm")
plt.xlabel("News Type")
plt.ylabel("Count")
plt.title("Distribution of Fake vs Real News")
plt.show()

# Show credibility by source
plt.figure(figsize=(12, 6))
source_counts = df.groupby('Source')['Label'].value_counts().unstack()
source_counts.plot(kind="bar", stacked=True, colormap="Paired", figsize=(12, 6))
plt.title("Fake vs Real News by Source")
plt.xlabel("Source")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(["Fake", "Real"])
plt.show()
