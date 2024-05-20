from data_preparation import load_and_preprocess_data, split_data
from model_definition import create_logistic_model
from local_training import local_train
from aggregation import decentralized_aggregation
from evaluation import evaluate_model
from utils import setup_logging

def main():
    logger = setup_logging()
    
    # Load and preprocess the dataset
    file_path = 'payment_fraud_dataset.csv'
    features, labels = load_and_preprocess_data(file_path)
    logger.info("Data loaded and preprocessed successfully.")
    
    # Split the dataset
    num_clients = 10
    data_split = split_data(features, labels, num_clients)
    logger.info(f"Data split into {num_clients} parts.")
    
    # Federated learning process
    num_rounds = 5
    client_models = [create_logistic_model() for _ in range(num_clients)]
    
    for round_num in range(num_rounds):
        # Local training
        logger.info(f"Starting round {round_num + 1} of local training.")
        client_models = [local_train(client_data, model) for client_data, model in zip(data_split, client_models)]
        
        # Decentralized aggregation
        logger.info(f"Aggregating models after round {round_num + 1}.")
        client_models = decentralized_aggregation(client_models)
        
        logger.info(f'Round {round_num + 1} completed')
    
    # Evaluate the final model
    logger.info("Evaluating the final model.")
    evaluate_model(client_models[0], file_path)

if __name__ == "__main__":
    main()
