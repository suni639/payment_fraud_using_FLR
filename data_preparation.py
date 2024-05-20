import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    data = pd.read_csv(file_path)
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    
    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features, labels

def split_data(features, labels, num_clients):
    """Split data into parts for each client."""
    data_split = []
    data_size = len(labels)
    split_size = data_size // num_clients
    
    for i in range(num_clients):
        start = i * split_size
        end = (i + 1) * split_size if i != num_clients - 1 else data_size
        client_data = (features[start:end], labels[start:end])
        data_split.append(client_data)
    
    return data_split
