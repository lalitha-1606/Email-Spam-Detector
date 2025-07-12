import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels: 'ham' = 0, 'spam' = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Convert text data to numerical vectors using TF-IDF
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Test with your own message
while True:
    user_input = input("\nEnter a message to check (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    # Preprocess and transform the input
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)

    if prediction[0] == 1:
        print("💥 This is a SPAM message.")
    else:
        print("✅ This is a HAM (not spam) message.")
import pickle

# Save the trained model
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)


