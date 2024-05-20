from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def local_train(data, model):
    """Train the model on local client data and capture training metrics."""
    try:
        features, labels = data
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
        model.fit(X_train, y_train)
        
        # Capture training accuracy
        train_preds = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        print(f"Training accuracy: {train_accuracy}")
        
        return model
    except Exception as e:
        print(f"Error during local training: {e}")
        return None
