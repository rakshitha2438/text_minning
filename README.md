# text_minning
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Load Dataset
# Assuming the dataset has two columns: "text" (social network posts) and "sentiment" (labels: positive/negative/neutral)
data = pd.read_csv("social_network_posts.csv")  # Replace with your dataset path
print(data.head())

# 2. Data Preprocessing
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))  # Remove stopwords
    return text

data['cleaned_text'] = data['text'].apply(clean_text)
print("\nCleaned Text Sample:\n", data['cleaned_text'].head())

# 3. Split Data
X = data['cleaned_text']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Text Vectorization (Bag of Words)
vectorizer = CountVectorizer(max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train).toarray()
X_test_vectors = vectorizer.transform(X_test).toarray()

# 5. Train Sentiment Classifier (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test_vectors)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. Visualize Results
# Word Cloud of Positive and Negative Words
positive_texts = ' '.join(data[data['sentiment'] == 'positive']['cleaned_text'])
negative_texts = ' '.join(data[data['sentiment'] == 'negative']['cleaned_text'])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
wordcloud_positive = WordCloud(width=400, height=200, background_color='white').generate(positive_texts)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title("Positive Words")
plt.axis('off')

plt.subplot(1, 2, 2)
wordcloud_negative = WordCloud(width=400, height=200, background_color='black').generate(negative_texts)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title("Negative Words")
plt.axis('off')

plt.tight_layout()
plt.show()

# 8. Save Model and Vectorizer
import joblib
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
