# Using Federated Logistic Regression to Detect Fraudulent Payments
This project aims to demonstrate the implementation of a decentralized federated learning system, using a simple logistic regression, to detect online payments fraud. The dataset (downloaded from ![Kaggle](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset/data)) is split across 20 clients, each of which trains a local model on their data. The models are then aggregated in a decentralized manner to form a global model.

## NOTE:
This is the second iteration of this project, with the first iteration using a complex and computationally expensive neural network, which failed to execute. This happened for the following reasons:
- high number of neurons and layers
- large batch sizes that used up too much memory and computational load
- high number of epochs thereby increasing training time
- use of heavy weight frameworks (TensorFlow)

**As the aim of the project was to gain a better understanding of the mechanics behind federated learning, the nodes/clients in this project use Logistic Regression instead of Neural Networks.**  

## Project Structure
   
```css
federated_learning_project/
│
├── data_preparation.py
├── model_definition.py
├── local_training.py
├── aggregation.py
├── evaluation.py
├── utils.py
├── main.py
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt
```
### File Descriptions

- `data_preparation.py`: Functions for loading, preprocessing, and splitting the dataset.
- `model_definition.py`: Defines the logistic regression model.
- `local_training.py`: Implements local training logic for each client.
- `aggregation.py`: Contains the logic for decentralized aggregation of model updates.
- `evaluation.py`: Includes functions to evaluate the final aggregated model.
- `utils.py`: Provides utility functions such as setting up logging.
- `main.py`: The main script that integrates all modules and runs the federated learning process.

## Installation

### Prerequisites

- Python 3.6 or higher
- Pip (Python package installer)

## Instructions to Run the Project
1. Download or clone the repository to your local machine.
2. Navigate to the project directory in your terminal or PowerShell.
3. Install the required dependencies using the provided requirements.txt file:
```powershell
pip install -r requirements.txt
```
4. Ensure the dataset is located in the project root
5. Run the main script to start the federated learning process:
```powershell
python main.py
```
### Contributions
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.