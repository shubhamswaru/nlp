import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_text(text):
    """
    Preprocess text by removing non-alphabetical characters and converting to lowercase.
    """
    return re.sub(r'[^a-z\s]', '', text.lower())

def load_and_preprocess_data(pos_file, neg_file):
    """
    Load and preprocess positive and negative reviews from files.
    """
    with open(pos_file, 'r', encoding='latin-1') as f:
        positive_reviews = [preprocess_text(review) for review in f.readlines()]
    
    with open(neg_file, 'r', encoding='latin-1') as f:
        negative_reviews = [preprocess_text(review) for review in f.readlines()]
    
    return positive_reviews, negative_reviews

def split_data(positive_reviews, negative_reviews):
    
    """
    Split data into training, validation, and test sets.
    """
    train_pos = positive_reviews[:4000]
    train_neg = negative_reviews[:4000]
    val_pos = positive_reviews[4000:4500]
    val_neg = negative_reviews[4000:4500]
    test_pos = positive_reviews[4500:]
    test_neg = negative_reviews[4500:]
    
    train_texts = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)
    val_texts = val_pos + val_neg
    val_labels = [1] * len(val_pos) + [0] * len(val_neg)
    test_texts = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def vectorize_text(train_texts, val_texts, test_texts):
    """
    Vectorize text data using TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_val, X_test, vectorizer

def tune_hyperparameters(X_train, y_train):
    """
    Tune hyperparameters for Logistic Regression using GridSearchCV.
    """
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2'],
        'max_iter': [100, 200, 500]
    }
    
    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_, grid_search.best_estimator_

def evaluate_model_with_cross_validation(model, X_train, y_train):
    """
    Evaluate model using cross-validation and return the average accuracy score.
    """
    cv_scores = cross_val_score(model, X_train, y_train, cv=3)
    return cv_scores.mean()

def generate_classification_report_with_counts(model, X_val, y_val, X_test, y_test):
    """
    Generate classification report, confusion matrix, and print counts of TP, TN, FP, FN samples.
    Also, calculate Precision, Recall, and F1-Score.
    """
    test_predictions = model.predict(X_test)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, test_predictions)
    
    # Count True Positives, True Negatives, False Positives, and False Negatives
    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]
    
    # Calculate Precision, Recall, and F1-Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print counts and metrics
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1_score}")
    
    # Return other classification metrics for further reporting
    val_accuracy = accuracy_score(y_val, model.predict(X_val))
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    classification_rep = classification_report(y_test, test_predictions)
    
    return val_accuracy, test_accuracy, conf_matrix, classification_rep

def visualize_confusion_matrix(conf_matrix):
    """
    Visualize confusion matrix using Seaborn heatmap.
    """
    plt.figure(figsize=(6,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def main():
    """
    Main function to run the sentiment analysis model, tune hyperparameters, and evaluate the model.
    """
    # Load and preprocess data
    positive_reviews, negative_reviews = load_and_preprocess_data('rt-polarity.pos', 'rt-polarity.neg')
    
    # Split data into train, validation, and test sets
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data(positive_reviews, negative_reviews)
    
    # Vectorize text data
    X_train, X_val, X_test, vectorizer = vectorize_text(train_texts, val_texts, test_texts)
    
    # Tune hyperparameters
    best_params, best_lr = tune_hyperparameters(X_train, train_labels)
    print(f"Best parameters found: {best_params}")
    
    # Evaluate model with cross-validation
    cv_accuracy = evaluate_model_with_cross_validation(best_lr, X_train, train_labels)
    print(f"Cross-Validation Accuracy: {cv_accuracy}")
    
    # Generate classification report and print counts of TP, TN, FP, FN samples along with Precision, Recall, and F1-Score
    val_accuracy, test_accuracy, conf_matrix, classification_rep = generate_classification_report_with_counts(
        best_lr, X_val, val_labels, X_test, test_labels
    )
    
    # Print classification results
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    print("Classification Report:\n", classification_rep)
    
    # Visualize confusion matrix
    visualize_confusion_matrix(conf_matrix)

if __name__ == '__main__':
    main()
