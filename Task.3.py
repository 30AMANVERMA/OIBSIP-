import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset and specify the encoding
data = pd.read_csv('C:\\Users\\amanv\\Downloads\\spam.csv', encoding='ISO-8859-1')

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)

# Step 3: Tokenize and vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 4: Create and train the model
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Step 5: Predict on the test set
y_pred = classifier.predict(X_test_tfidf)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate a classification report for more detailed evaluation
print(classification_report(y_test, y_pred))
