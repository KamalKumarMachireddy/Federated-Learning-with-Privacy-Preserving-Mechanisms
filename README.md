# Federated Learning with Privacy-Preserving Mechanisms

A privacy-preserving federated learning system that allows multiple clients to collaboratively train a shared machine learning model without exposing their raw data. This implementation includes mechanisms such as differential privacy, secure aggregation, and a real-time monitoring dashboard to track training progress and privacy metrics.

---

[Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/code/youssefelbadry10/credit-card-fraud-detection/input)


## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tools and Libraries](#tools-and-libraries)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Overview

Federated Learning is a distributed machine learning paradigm where multiple clients collaboratively train a shared global model under the orchestration of a central server—without sharing their private data.

This project focuses on building a **secure and scalable** federated learning system with:

- **Privacy-Preserving Mechanisms**: Differential privacy ensures that individual client data cannot be reverse-engineered. Secure aggregation protects update confidentiality using homomorphic encryption.
- **Real-Time Monitoring Dashboard**: Visualizes training metrics like global accuracy, privacy budget consumption, and client-level statistics.
- **Scalability**: Designed to support large numbers of clients across diverse applications.

> Use cases include fraud detection, healthcare analytics, predictive maintenance in IoT, and more.

---

## Key Features

- **Differential Privacy**: Uses gradient clipping and noise injection to ensure data confidentiality.
- **Secure Aggregation**: Utilizes homomorphic encryption (e.g., Paillier) to securely combine updates.
- **Central Server**: Coordinates federated learning rounds and manages the global model.
- **Clients**: Train local models and send encrypted updates to the server.
- **Monitoring Dashboard**: Live dashboard to monitor training progress and privacy metrics.
- **Use Case Included**: Example pipeline for credit card fraud detection.

---

## Tools and Libraries

- **Python**: Core language
- **TensorFlow**: Model training and evaluation
- **gRPC**: Communication between server and clients
- **Dash & Plotly**: Real-time interactive dashboard
- **Pandas/Numpy**: Data manipulation and processing
- **python-paillier** *(optional)*: Homomorphic encryption library

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- Ensure the dataset (`fraud_data.csv`) is available in the `data/` directory or use your own.

### Installation

Clone the repository:

```bash
git clone https://github.com/your-repo/federated-learning.git
cd federated-learning

```

### Federated Learning Project Setup

#### Install Dependencies

```bash
pip install -r requirements.txt
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. federated_learning.proto

```
## Usage
### Start the Central Server
``` bash
python central_server.py

```
### Start the Monitoring Dashboard
``` bash
python dashboard.py

```

[You can just navigate to http://localhost:8050 in your browser.](http://localhost:8050)


### Start Clients
Run this command for each client (in separate terminals):

``` bash
python client.py --client_id client_1 --server localhost:50051 --data_path ./data/fraud_data.csv --rounds 5 --epochs 3
Note: Change the --client_id for each additional client (e.g., client_2, client_3, etc.).

```

### Project Structure
``` graphql
project_root/
├── central_server.py           # Central server logic
├── client.py                   # Client logic
├── dashboard.py                # Real-time dashboard
├── federated_learning.proto    # gRPC service definition
├── federated_pb2.py            # Generated gRPC Python code
├── federated_pb2_grpc.py       # Generated gRPC service classes
├── data/
│   └── fraud_data.csv          # Example fraud detection dataset
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation

```


    
