import numpy as np

def decentralized_aggregation(client_models):
    """Aggregate client models by averaging their coefficients."""
    n_clients = len(client_models)
    avg_model = client_models[0]
    
    coef = np.mean([model.coef_ for model in client_models], axis=0)
    intercept = np.mean([model.intercept_ for model in client_models], axis=0)
    
    avg_model.coef_ = coef
    avg_model.intercept_ = intercept
    
    return [avg_model] * n_clients
