import pandas as pd
import spacy
import numpy as np
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os

# Function to load datasets
def load_datasets():
    """
    Load the job postings datasets and filter out fraudulent ones from the second dataset.

    Args:
        None

    Returns:
        df_fake_postings (pd.DataFrame): DataFrame containing the fake job postings.
        df_fake_real_postings (pd.DataFrame): DataFrame containing only the real job postings 
                                              (fraudulent ones are removed).
    """
    df_fake_postings = pd.read_csv('/Users/farhathanwar/Documents/Code/AI-Job-Posting-Classifier/DATA/FakePostings_10k.csv')
    df_fake_real_postings = pd.read_csv('/Users/farhathanwar/Documents/Code/AI-Job-Posting-Classifier/DATA/job_postings_18k.csv')
    df_fake_real_postings = df_fake_real_postings[df_fake_real_postings['fraudulent'] == 0]
    return df_fake_postings, df_fake_real_postings

# Function to preprocess text using spaCy
def preprocess_text_spacy(text, nlp):
    """
    Preprocess the input text by applying tokenization, lemmatization, and stop word removal using spaCy.

    Args:
        text (str): The text to preprocess.
        nlp (spacy.language): The spaCy language model to use for processing.

    Returns:
        str: The processed text with lemmatized tokens, stop words removed, and only alphabetic words retained.
    """
    if isinstance(text, str):
        doc = nlp(text)
        processed_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(processed_words)
    else:
        return ""  # Return empty string if text is not valid (e.g., NaN or float)

# Function to preprocess the datasets
def preprocess_data(df_fake_postings, df_fake_real_postings, columns_to_process, nlp):
    """
    Apply text preprocessing to multiple columns in both dataframes using spaCy.

    Args:
        df_fake_postings (pd.DataFrame): DataFrame containing the fake job postings.
        df_fake_real_postings (pd.DataFrame): DataFrame containing both fake and real job postings.
        columns_to_process (list): List of columns (strings) that need text preprocessing.
        nlp (spacy.language): The spaCy language model to use for processing.

    Returns:
        df_fake_postings (pd.DataFrame): DataFrame with processed text columns.
        df_fake_real_postings (pd.DataFrame): DataFrame with processed text columns.
    """
    for column in columns_to_process:
        df_fake_postings[f'processed_{column}'] = df_fake_postings[column].apply(lambda x: preprocess_text_spacy(x, nlp))
        df_fake_real_postings[f'processed_{column}'] = df_fake_real_postings[column].apply(lambda x: preprocess_text_spacy(x, nlp))
    return df_fake_postings, df_fake_real_postings

# Function to combine text features into one column
def combine_text_features(df_fake_postings, df_fake_real_postings, columns_to_process):
    """
    Combine multiple text columns into a single column by joining the processed text data.

    Args:
        df_fake_postings (pd.DataFrame): DataFrame containing the fake job postings.
        df_fake_real_postings (pd.DataFrame): DataFrame containing both fake and real job postings.
        columns_to_process (list): List of columns to combine.

    Returns:
        df_fake_postings (pd.DataFrame): DataFrame with combined text column.
        df_fake_real_postings (pd.DataFrame): DataFrame with combined text column.
    """
    df_fake_postings['combined_text'] = df_fake_postings[columns_to_process].fillna('').apply(lambda x: ' '.join(x), axis=1)
    df_fake_real_postings['combined_text'] = df_fake_real_postings[columns_to_process].fillna('').apply(lambda x: ' '.join(x), axis=1)
    return df_fake_postings, df_fake_real_postings

# Function to fit and transform TF-IDF vectorizer
def fit_tfidf(df_fake_postings, df_fake_real_postings):
    """
    Fit a TF-IDF vectorizer on the text data and transform both the fake and real job postings.

    Args:
        df_fake_postings (pd.DataFrame): DataFrame containing the fake job postings.
        df_fake_real_postings (pd.DataFrame): DataFrame containing both fake and real job postings.

    Returns:
        tfidf (TfidfVectorizer): Fitted TF-IDF vectorizer.
        tfidf_fake (sparse matrix): Transformed TF-IDF representation for fake job postings.
        tfidf_real (sparse matrix): Transformed TF-IDF representation for real job postings.
    """
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
    tfidf_fake = tfidf.fit_transform(df_fake_postings['combined_text'])
    tfidf_real = tfidf.transform(df_fake_real_postings['combined_text'])
    return tfidf, tfidf_fake, tfidf_real

# Function to generate sentence embeddings
def sentence_to_embedding(sentence, model):
    """
    Convert a sentence to a vector representation (embedding) using a pre-trained word embedding model.

    Args:
        sentence (str): The sentence to convert into an embedding.
        model (gensim.models.keyedvectors.KeyedVectors): Pre-trained word embeddings model (e.g., GloVe).

    Returns:
        np.ndarray: The vector representation of the sentence (averaged word embeddings).
    """
    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# Function to apply sentence embeddings
def apply_sentence_embeddings(df_fake_postings, df_fake_real_postings, glove_model):
    """
    Apply sentence embeddings to the combined text column in both dataframes using a pre-trained GloVe model.

    Args:
        df_fake_postings (pd.DataFrame): DataFrame containing the fake job postings.
        df_fake_real_postings (pd.DataFrame): DataFrame containing both fake and real job postings.
        glove_model (gensim.models.keyedvectors.KeyedVectors): Pre-trained GloVe word embeddings model.

    Returns:
        df_fake_postings (pd.DataFrame): DataFrame with added sentence embeddings.
        df_fake_real_postings (pd.DataFrame): DataFrame with added sentence embeddings.
    """
    df_fake_postings['sentence_embedding'] = df_fake_postings['combined_text'].apply(lambda x: sentence_to_embedding(x, glove_model))
    df_fake_real_postings['sentence_embedding'] = df_fake_real_postings['combined_text'].apply(lambda x: sentence_to_embedding(x, glove_model))
    return df_fake_postings, df_fake_real_postings

# Function to preprocess structured features
def preprocess_structured_features(data, columns):
    """
    One-hot encode structured features in the dataset.

    Args:
        data (pd.DataFrame): The dataset containing structured features.
        columns (list): List of categorical columns to encode.

    Returns:
        encoded_array (np.ndarray): One-hot encoded matrix of structured features.
        encoder (OneHotEncoder): Fitted OneHotEncoder used for encoding.
    """
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_array = encoder.fit_transform(data[columns].fillna('Unknown'))
    return encoded_array, encoder

# Function to train and evaluate models for missing feature prediction
def infer_missing_feature(data, target, features, tfidf=None, max_features=5000):
    """
    Train a classification model to predict missing values for a given target feature.

    Args:
        data (pd.DataFrame): DataFrame containing both fake and real job postings for training.
        target (str): The target feature (e.g., 'has_company_logo', 'has_questions', 'telecommuting').
        features (list): List of features to use for training.
        tfidf (TfidfVectorizer, optional): Pretrained TF-IDF vectorizer for text features. Defaults to None.
        max_features (int): Maximum number of features for the TF-IDF vectorizer. Defaults to 5000.

    Returns:
        model (RandomForestClassifier): Trained RandomForest model for predicting the target.
        tfidf (TfidfVectorizer): The fitted TF-IDF vectorizer.
        report (str): Classification report for model evaluation.
        encoder (OneHotEncoder): OneHotEncoder used for encoding categorical features.
    """
    clean_data = data.dropna(subset=[target])
    X = clean_data[features].fillna('')
    y = clean_data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_columns = ['employment_type', 'industry']
    structured_train, encoder = preprocess_structured_features(X_train, categorical_columns)
    structured_test = encoder.transform(X_test[categorical_columns].fillna('Unknown'))
    
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        X_train_tfidf = tfidf.fit_transform(X_train['processed_description'])
    else:
        X_train_tfidf = tfidf.transform(X_train['processed_description'])
        
    X_test_tfidf = tfidf.transform(X_test['processed_description'])
    
    X_train_combined = np.hstack([X_train_tfidf.toarray(), structured_train])
    X_test_combined = np.hstack([X_test_tfidf.toarray(), structured_test])
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_combined, y_train)
    
    y_pred = model.predict(X_test_combined)
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_combined)[:, 1])
    
    print(f"Classification Report for {target}:\n{report}")
    print(f"AUC Score: {auc:.2f}")
    
    return model, tfidf, report, encoder

# Function to apply prediction for missing data
def apply_prediction_for_missing_data(df_fake_postings, df_fake_real_postings, target, features, tfidf):
    """
    A function to predict missing data for a given target using the infer_missing_feature pipeline.
    
    Args:
        df_fake_postings: DataFrame containing fake job postings with potential missing target values.
        df_fake_real_postings: DataFrame containing both fake and real job postings for training.
        target: The target feature to predict (e.g., 'has_company_logo', 'has_questions', 'telecommuting').
        features: List of features to be used in the prediction model.
        tfidf: Fitted TF-IDF vectorizer.
    
    Returns:
        df_fake_postings: DataFrame with predicted values for the target feature.
    """
    logo_model, _, _, encoder = infer_missing_feature(
        data=df_fake_real_postings, 
        target=target, 
        features=features, 
        tfidf=tfidf
    )

    X_fake = df_fake_postings[features].fillna('')
    structured_fake = encoder.transform(X_fake[['employment_type', 'industry']].fillna('Unknown'))
    X_fake_tfidf = tfidf.transform(X_fake['processed_description'])
    X_fake_combined = np.hstack([X_fake_tfidf.toarray(), structured_fake])

    df_fake_postings[target] = logo_model.predict(X_fake_combined)
    return df_fake_postings

# Function to encode categorical features and align columns
def encode_and_align_categorical_features(df_fake_postings, df_fake_real_postings, column_name):
    """
    Encodes the specified categorical column in both datasets using one-hot encoding,
    and aligns the columns between the two datasets to ensure they have the same set of features.

    Args:
        df_fake_postings (pd.DataFrame): DataFrame containing the fake postings data.
        df_fake_real_postings (pd.DataFrame): DataFrame containing the real postings data.
        column_name (str): The name of the categorical column to be encoded.

    Returns:
        tuple: A tuple containing two DataFrames:
            - df_fake_postings (pd.DataFrame): The modified fake postings DataFrame with encoded columns.
            - df_fake_real_postings (pd.DataFrame): The modified real postings DataFrame with encoded columns.
    """
    # One-hot encode the column in both datasets
    df_fake_encoded = pd.get_dummies(df_fake_postings[column_name], prefix=column_name, dummy_na=False)
    df_real_encoded = pd.get_dummies(df_fake_real_postings[column_name], prefix=column_name, dummy_na=False)
    
    # Align columns to ensure both dataframes have the same set of features
    df_fake_encoded, df_real_encoded = df_fake_encoded.align(df_real_encoded, join='outer', axis=1, fill_value=0)
    
    # Drop the original column and add the encoded columns
    df_fake_postings = df_fake_postings.drop(columns=[column_name], errors='ignore').join(df_fake_encoded)
    df_fake_real_postings = df_fake_real_postings.drop(columns=[column_name], errors='ignore').join(df_real_encoded)
    
    return df_fake_postings, df_fake_real_postings

# Function to combine data and prepare for classification
def prepare_classification_data(df_fake_postings, df_fake_real_postings, tfidf_fake, tfidf_real):
    """
    Combine text features and structured features from both fake and real job postings for classification.

    Args:
        df_fake_postings (pd.DataFrame): DataFrame containing the fake job postings.
        df_fake_real_postings (pd.DataFrame): DataFrame containing both fake and real job postings.
        tfidf_fake (sparse matrix): TF-IDF representation of fake job postings.
        tfidf_real (sparse matrix): TF-IDF representation of real job postings.

    Returns:
        X (np.ndarray): Combined feature matrix for both fake and real job postings.
        y (np.ndarray): Labels for classification (1 for fake, 0 for real).
    """

    # Dynamically collect all one-hot encoded employment_type columns
    employment_type_columns = [col for col in df_fake_postings.columns if col.startswith('employment_type_')]

    print("Employment Type Columns Detected:", employment_type_columns)
    
    # Add these to the structured features
    structured_features = ['telecommuting', 'has_company_logo', 'has_questions'] + employment_type_columns
    
    fake_structured = df_fake_postings[structured_features].values
    real_structured = df_fake_real_postings[structured_features].values

    X_fake = np.hstack([tfidf_fake.toarray(), fake_structured])
    X_real = np.hstack([tfidf_real.toarray(), real_structured])

    X = np.vstack([X_fake, X_real])
    y = np.array([1] * X_fake.shape[0] + [0] * X_real.shape[0])

    return X, y

# Function to perform train-test split
def perform_train_test_split(X, y):
    """
    Perform a train-test split on the given feature matrix and labels.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.

    Returns:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Test features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Test labels.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate baseline models
def train_baseline_models(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates baseline machine learning models on the given training and testing data.
    The models evaluated include Naive Bayes, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).

    Args:
        X_train (pd.DataFrame or np.ndarray): The feature matrix for the training data.
        X_test (pd.DataFrame or np.ndarray): The feature matrix for the testing data.
        y_train (pd.Series or np.ndarray): The target labels for the training data.
        y_test (pd.Series or np.ndarray): The target labels for the testing data.

    Returns:
        dict: A dictionary containing the evaluation results for each model:
            - 'Naive Bayes': A dictionary with accuracy, classification report, and ROC AUC score.
            - 'SVM': A dictionary with accuracy, classification report, and ROC AUC score.
            - 'KNN': A dictionary with accuracy, classification report, and a note for ROC AUC (since KNN does not support probabilities by default).
    """
    
    # Store results
    results = {}
    
    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_predictions = nb_model.predict(X_test)
    nb_auc = roc_auc_score(y_test, nb_model.predict_proba(X_test)[:, 1])
    results['Naive Bayes'] = {
        'accuracy': accuracy_score(y_test, nb_predictions),
        'classification_report': classification_report(y_test, nb_predictions),
        'roc_auc': nb_auc
    }
    
    # SVM (with linear kernel)
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_auc = roc_auc_score(y_test, svm_model.decision_function(X_test))
    results['SVM'] = {
        'accuracy': accuracy_score(y_test, svm_predictions),
        'classification_report': classification_report(y_test, svm_predictions),
        'roc_auc': svm_auc
    }
    
    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    results['KNN'] = {
        'accuracy': accuracy_score(y_test, knn_predictions),
        'classification_report': classification_report(y_test, knn_predictions),
        'roc_auc': 'N/A (KNN does not support probability output by default)'
    }
    
    return results

# Main function
def main():
    # Load datasets
    df_fake_postings, df_fake_real_postings = load_datasets()
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Preprocess data
    columns_to_process = ['title', 'description', 'company_profile', 'requirements', 'benefits']
    df_fake_postings, df_fake_real_postings = preprocess_data(df_fake_postings, df_fake_real_postings, columns_to_process, nlp)
    
    # Combine text features
    df_fake_postings, df_fake_real_postings = combine_text_features(df_fake_postings, df_fake_real_postings, columns_to_process)
    
    # Fit TF-IDF once and reuse it
    tfidf, tfidf_fake, tfidf_real = fit_tfidf(df_fake_postings, df_fake_real_postings)
    
    # Load pretrained GloVe model for embeddings
    glove_model = api.load('glove-wiki-gigaword-50')
    
    # Apply sentence embeddings (optional, if embeddings are used for predictions)
    df_fake_postings, df_fake_real_postings = apply_sentence_embeddings(df_fake_postings, df_fake_real_postings, glove_model)
    
    # Predict missing features (pass fitted TF-IDF for consistent feature extraction)
    features = ['processed_description', 'employment_type', 'industry', 'requirements']
    
    df_fake_postings = apply_prediction_for_missing_data(
        df_fake_postings, df_fake_real_postings, target='has_company_logo', features=features, tfidf=tfidf
    )
    df_fake_postings = apply_prediction_for_missing_data(
        df_fake_postings, df_fake_real_postings, target='has_questions', features=features, tfidf=tfidf
    )
    df_fake_postings = apply_prediction_for_missing_data(
        df_fake_postings, df_fake_real_postings, target='telecommuting', features=features, tfidf=tfidf
    )
    
    # Encode and align 'employment_type'
    df_fake_postings, df_fake_real_postings = encode_and_align_categorical_features(
        df_fake_postings, df_fake_real_postings, 'employment_type'
    )

    print("df_fake_postings columns:", df_fake_postings.columns)
    print("df_fake_real_postings columns:", df_fake_real_postings.columns)

    
    # Prepare data for classification
    structured_features = ['telecommuting', 'has_company_logo', 'has_questions'] + \
                          [col for col in df_fake_postings.columns if 'employment_type_' in col]
    fake_structured = df_fake_postings[structured_features].values
    real_structured = df_fake_real_postings[structured_features].values

    X, y = prepare_classification_data(df_fake_postings, df_fake_real_postings, tfidf_fake, tfidf_real)
    X_train, X_test, y_train, y_test = perform_train_test_split(X, y)

    # Train and evaluate baseline models
    results = train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Print results
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Updated df_fake_postings columns:\n{df_fake_postings.columns.tolist()}")
    
    for model, metrics in results.items():
        print(f"\nResults for {model}:")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"ROC AUC: {metrics['roc_auc']}")
        print(f"Classification Report:\n{metrics['classification_report']}")


if __name__ == "__main__":
    main()

