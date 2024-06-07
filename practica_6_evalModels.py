import csv
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import spacy

# Load spaCy model
nlp = spacy.load("es_core_news_sm")

# Define Preprocessing Function using spaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    return ' '.join(tokens)

# Load Data using csv library
data = []
with open('Rest_Mex_2022.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    i = 0
    for row in reader:
        title_opinion = row['Title'] + ' ' + row['Opinion']
        processed_text = preprocess_text(title_opinion)
        data.append([processed_text, row['Polarity']])
        i+=1
        print(f"Proccessed: {i}")

# Convert to DataFrame
df = pd.DataFrame(data, columns=['text', 'Polarity'])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['Polarity'], test_size=0.2, random_state=0)

# Text Representations
vectorizers = {
    'Binarized': CountVectorizer(binary=True),
    'Frequency': CountVectorizer(),
    'TF-IDF': TfidfVectorizer(),
}

# Model Definitions
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
}

# Training and Evaluation
best_score = 0
best_pipeline = None
for vec_name, vectorizer in vectorizers.items():
    for model_name, model in models.items():
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])
        cv_results = cross_validate(pipeline, X_train, y_train, cv=5, scoring='f1_macro')
        mean_score = cv_results['test_score'].mean()
        print(f'{vec_name} + {model_name}: {mean_score}')
        if mean_score > best_score:
            best_score = mean_score
            best_pipeline = pipeline

# Train the selected model on the full training set
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

# Evaluation
print('Best pipeline:', best_pipeline)
print('Test set average f1_macro:', f1_score(y_test, y_pred, average='macro'))
print('Classification report:\n', classification_report(y_test, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
