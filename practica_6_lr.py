import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import spacy
from nltk.corpus import stopwords
import nltk
from imblearn.pipeline import Pipeline as ImbPipeline

# Download NLTK data
nltk.download('stopwords')

# Load Spacy model
nlp = spacy.load('es_core_news_sm')

# Define stopwords
stop_words = set(stopwords.words('spanish'))

def load_lexicon(filepath):
    """
    Loads the lexicon from a CSV file and returns it as a dictionary.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        dict: A dictionary containing the lexicon.
    """
    lexicon = pd.read_csv(filepath, encoding='utf-8')
    lexicon_dict = {}
    for _, row in lexicon.iterrows():
        lexicon_dict[row['Palabra']] = {
            'Nula': row['Nula[%]'],
            'Baja': row['Baja[%]'],
            'Media': row['Media[%]'],
            'Alta': row['Alta[%]'],
            'PFA': row['PFA'],
            'Categoria': row['Categoría']
        }
    return lexicon_dict

def lexicon_features(text, lexicon_dict):
    """
    Extracts lexicon features from the given text using the provided lexicon dictionary.

    Args:
        text (str): The input text.
        lexicon_dict (dict): A dictionary containing the lexicon.

    Returns:
        tuple: A tuple containing the lexicon features.
    """
    words = text.split()
    nula, baja, media, alta, pfa, categoria = 0, 0, 0, 0, 0.0, ""
    for word in words:
        if word in lexicon_dict:
            nula += lexicon_dict[word]['Nula']
            baja += lexicon_dict[word]['Baja']
            media += lexicon_dict[word]['Media']
            alta += lexicon_dict[word]['Alta']
            pfa += lexicon_dict[word]['PFA']
            categoria += lexicon_dict[word]['Categoria'] if lexicon_dict[word]['Categoria'] != 'nan' else ''
    total_words = len(words)
    if total_words > 0:
        nula /= total_words
        baja /= total_words
        media /= total_words
        alta /= total_words
        pfa /= total_words
    return nula, baja, media, alta, pfa, categoria

def preprocess_text(text, lexicon_dict):
    """
    Preprocesses the given text by applying various text preprocessing techniques.

    Args:
        text (str): The input text.
        lexicon_dict (dict): A dictionary containing the lexicon.

    Returns:
        tuple: A tuple containing the preprocessed text and lexicon features.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    #text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    #text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatization
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc])
    # Extract lexicon features
    nula, baja, media, alta, pfa, categoria = lexicon_features(text, lexicon_dict)
    
    # Agregar manejo de negacion
    
    
    
    return text, nula, baja, media, alta, pfa, categoria

def preprocess_dataframe(df, lexicon_dict):
    """
    Preprocesses the dataframe by applying text preprocessing to the 'Title' and 'Opinion' columns.
    It uses the lexicon dictionary to extract features from the text.

    Args:
        df (pandas.DataFrame): The input dataframe.
        lexicon_dict (dict): A dictionary containing the lexicon for text preprocessing.

    Returns:
        pandas.DataFrame: The preprocessed dataframe with additional columns for the processed text and categories.
    """
    features = df.apply(lambda row: preprocess_text(str(row['Title']) + ' ' + str(row['Opinion']), lexicon_dict), axis=1)
    df['text'] = features.apply(lambda x: x[0])
    df['nula'] = features.apply(lambda x: x[1])
    df['baja'] = features.apply(lambda x: x[2])
    df['media'] = features.apply(lambda x: x[3])
    df['alta'] = features.apply(lambda x: x[4])
    df['pfa'] = features.apply(lambda x: x[5])
    df['Categoría'] = features.apply(lambda x: x[6])
    return df

def main():
    # Define the path to the corpus and lexicon
    corpus = 'Rest_Mex_2022.xlsx'
    lexicon = 'SEL_full.csv'

    # Load data and lexicon, preprocess data, and verify if the data is available in a pickle file
    try:
        data = pd.read_pickle(f'proccessed_dataframe_{corpus}.pkl')
        print("Data loaded successfully.")
        print(data.head())
    except FileNotFoundError:
        data = pd.read_excel(corpus)
        print("Data loaded successfully.")
        print(data.head())

        lexicon_dict = load_lexicon(lexicon)
        print("Lexicon loaded successfully.")

        data = preprocess_dataframe(data, lexicon_dict)
        print("Data preprocessed successfully.")
        print(data.head())

        data.to_pickle(f'proccessed_dataframe_{corpus}.pkl')
        print("Data saved successfully.")


    # Ensure the expected columns are present
    expected_columns = ['text', 'nula', 'baja', 'media', 'alta', 'pfa']
    for column in expected_columns:
        if column not in data.columns:
            raise ValueError(f"Missing column: {column}")

    X = data[['text', 'nula', 'baja', 'media', 'alta', 'pfa']]
    y = data['Polarity']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

    # Define the pipeline with feature union
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer())
    ])

    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'text'),
            ('numeric', StandardScaler(), ['nula', 'baja', 'media', 'alta', 'pfa'])
        ]
    )

    # Integrate SMOTE into the pipeline using imbalanced-learn's Pipeline
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE()),
        ('classifier', LogisticRegression(C=3,solver='lbfgs', random_state=0, max_iter=10000, n_jobs=-1))
    ])

    # Cross-validation
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    print(f'Average F1 Macro Score (cross-validation): {scores.mean()}')

    # Train the model on full training data
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'Test F1 Macro Score: {f1}')
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()