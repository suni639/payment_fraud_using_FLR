from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, file_path):
    """Evaluate the model on the test dataset and capture evaluation metrics."""
    from data_preparation import load_and_preprocess_data
    
    features, labels = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    
    # Make predictions
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, average='binary')
    recall = recall_score(y_test, test_preds, average='binary')
    f1 = f1_score(y_test, test_preds, average='binary')
    
    # Log metrics
    print(f"Model evaluation metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
