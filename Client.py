import numpy as np
import tensorflow as tf
import grpc
import logging
import pickle
import time
import pandas as pd
from typing import List, Dict, Any
import os
import federated_learning_pb2
import federated_learning_pb2_grpc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClientDifferentialPrivacy:
    """
    Client-side implementation of differential privacy.
    """

    def __init__(self, epsilon=1.0, delta=1e-5, clip_threshold=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_threshold = clip_threshold
        logger.info(f"Initialized client DP with epsilon={epsilon}, delta={delta}")

    def apply_local_dp(self, gradients):
        """Apply differential privacy to local gradients"""
        logger.info("Applying local differential privacy to gradients")

        # Clip gradients to bound sensitivity
        clipped_gradients = self._clip_gradients(gradients)

        # Add noise calibrated to sensitivity and privacy parameters
        noisy_gradients = self._add_noise(clipped_gradients)

        return noisy_gradients

    def _clip_gradients(self, gradients):
        """Clip gradients to limit sensitivity"""
        clipped = []
        for grad in gradients:
            if isinstance(grad, np.ndarray):
                grad_norm = np.linalg.norm(grad)
                if grad_norm > self.clip_threshold:
                    grad = grad * (self.clip_threshold / grad_norm)
                clipped.append(grad)
            else:
                clipped.append(grad)
        return clipped

    def _add_noise(self, gradients):
        """Add Gaussian noise for differential privacy"""
        # Calculate noise scale based on privacy parameters
        noise_scale = np.sqrt(2 * np.log(1.25 / self.delta)) * self.clip_threshold / self.epsilon

        noisy_gradients = []
        for grad in gradients:
            if isinstance(grad, np.ndarray):
                noise = np.random.normal(0, noise_scale, grad.shape)
                noisy_gradients.append(grad + noise)
            else:
                noisy_gradients.append(grad)

        return noisy_gradients


class FraudDetectionPreprocessor:
    """
    Process the credit card fraud dataset.
    """

    def __init__(self):
        logger.info("Initializing FraudDetectionPreprocessor")

    def load_data(self, file_path):
        """Load the fraud detection dataset"""
        logger.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)

    def preprocess(self, df):
        """Preprocess the fraud detection data"""
        logger.info("Preprocessing fraud detection data")

        # Copy the dataframe to avoid modifying the original
        processed_df = df.copy()

        # Drop non-numerical columns or columns we don't want to use
        cols_to_drop = ['trans_date_trans_time', 'cc_num', 'first', 'last',
                        'street', 'city', 'state', 'zip', 'job', 'dob',
                        'trans_num', 'unix_timew']
        processed_df = processed_df.drop(columns=cols_to_drop, errors='ignore')

        # One-hot encode categorical columns
        categorical_cols = ['merchant', 'category', 'gender']
        for col in categorical_cols:
            if col in processed_df.columns:
                one_hot = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df, one_hot], axis=1)
                processed_df = processed_df.drop(columns=[col])

        # Normalize numerical columns
        numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
        for col in numerical_cols:
            if col in processed_df.columns:
                processed_df[col] = (processed_df[col] - processed_df[col].mean()) / processed_df[col].std()

        # Extract features and target
        X = processed_df.drop(columns=['is_fraud'], errors='ignore').values
        y = processed_df['is_fraud'].values if 'is_fraud' in processed_df.columns else None

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


class FederatedClient:
    """
    Client implementation for federated learning.
    """

    def __init__(self, client_id, server_address='localhost:50051'):
        self.client_id = client_id
        self.server_address = server_address
        self.dp = ClientDifferentialPrivacy()
        self.model = None
        self.current_round = 0
        logger.info(f"Initialized federated client {client_id} connecting to {server_address}")

    def connect_to_server(self):
        """Establish gRPC connection to the server"""
        logger.info(f"Connecting to server at {self.server_address}")
        # In a real implementation, use proper channel credentials
        channel = grpc.insecure_channel(self.server_address)
        # self.stub = federated_pb2_grpc.FederatedLearningStub(channel)
        logger.info(f"Connected to server")

    def get_global_model(self):
        """Fetch the current global model from the server"""
        logger.info("Requesting global model from server")

        # In a real implementation, use the generated stub
        # response = self.stub.GetGlobalModel(federated_pb2.GetModelRequest(
        #     client_id=self.client_id,
        #     version=self.current_round
        # ))

        # For demonstration purposes, mocking the response
        time.sleep(1)  # Simulate network delay

        # Placeholder for deserialized model weights
        # weights = pickle.loads(response.model_weights)
        # self.current_round = response.round

        # For now, create a dummy model if none exists
        if self.model is None:
            logger.info("Creating initial model")
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
        else:
            # In a real implementation, we would update the model weights
            logger.info(f"Received global model for round {self.current_round}")
            # self.model.set_weights(weights)

    def train_local_model(self, X_train, y_train, epochs=10, batch_size=32):
        """Train the model on local data"""
        logger.info(f"Training local model for {epochs} epochs")

        if self.model is None:
            logger.error("No model available for training")
            return None

        # Fit the model to local data
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        logger.info(f"Local training completed with loss={final_loss:.4f}, accuracy={final_accuracy:.4f}")

        return history

    def compute_model_update(self, original_weights=None):
        """
        Compute model update by comparing current model with original weights.

        If original_weights is None, return the current weights directly.
        """
        current_weights = self.model.get_weights()

        if original_weights is None:
            logger.info("No original weights provided, using current weights as update")
            return current_weights

        # Compute the difference (update)
        logger.info("Computing model update")
        updates = []
        for i in range(len(current_weights)):
            updates.append(current_weights[i] - original_weights[i])

        return updates

    def apply_differential_privacy(self, model_update):
        """Apply differential privacy to the model update"""
        logger.info("Applying differential privacy to model update")
        return self.dp.apply_local_dp(model_update)

    def submit_model_update(self, model_update, num_samples):
        """Send the model update to the server"""
        logger.info(f"Submitting model update for round {self.current_round}")

        # Serialize the model update
        serialized_update = pickle.dumps(model_update)

        # In a real implementation, use the generated stub
        # response = self.stub.SubmitUpdate(federated_pb2.ClientUpdate(
        #     client_id=self.client_id,
        #     round=self.current_round,
        #     model_update=serialized_update,
        #     num_samples=num_samples,
        #     training_loss=final_loss,
        #     training_accuracy=final_accuracy
        # ))

        # For demonstration purposes, mocking the response
        time.sleep(1)  # Simulate network delay
        success = True
        next_round = self.current_round + 1

        if success:
            logger.info(f"Model update accepted, next round: {next_round}")
            self.current_round = next_round
            return True
        else:
            logger.warning(f"Model update rejected")
            return False

    def evaluate_model(self, X_test, y_test):
        """Evaluate the current model on test data"""
        logger.info("Evaluating model on test data")

        if self.model is None:
            logger.error("No model available for evaluation")
            return None

        loss, accuracy, auc = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Model evaluation: loss={loss:.4f}, accuracy={accuracy:.4f}, AUC={auc:.4f}")

        return {'loss': loss, 'accuracy': accuracy, 'auc': auc}

    def federated_learning_round(self, X_train, y_train, X_test, y_test, epochs=5):
        """
        Execute one complete round of federated learning:
        1. Get the global model
        2. Train on local data
        3. Apply differential privacy to the update
        4. Submit the update to the server
        5. Evaluate the model
        """
        logger.info(f"Starting federated learning round {self.current_round}")

        # Get the global model
        self.get_global_model()

        # Keep a copy of the original weights for computing the update
        original_weights = self.model.get_weights()

        # Train the model on local data
        history = self.train_local_model(X_train, y_train, epochs=epochs)

        # Compute the model update
        model_update = self.compute_model_update(original_weights)

        # Apply differential privacy to the update
        private_update = self.apply_differential_privacy(model_update)

        # Submit the model update to the server
        success = self.submit_model_update(private_update, len(X_train))

        # Evaluate the updated model
        metrics = self.evaluate_model(X_test, y_test)

        return success, metrics, history


def main():
    """Main function to run a federated client"""
    import argparse

    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client_id', type=str, default=f'client_{os.getpid()}',
                        help='Unique client ID')
    parser.add_argument('--server', type=str, default='localhost:50051',
                        help='Server address in the format host:port')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the fraud detection dataset')
    parser.add_argument('--rounds', type=int, default=5,
                        help='Number of federated learning rounds to participate in')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of local training epochs per round')

    args = parser.parse_args()

    # Initialize preprocessor and load data
    preprocessor = FraudDetectionPreprocessor()
    try:
        df = preprocessor.load_data(args.data_path)
        X, y = preprocessor.preprocess(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        logger.info(
            f"Data loaded and preprocessed: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Initialize and connect the client
    client = FederatedClient(args.client_id, args.server)
    try:
        client.connect_to_server()
    except Exception as e:
        logger.error(f"Error connecting to server: {e}")
        return

    # Participate in multiple rounds of federated learning
    for round_num in range(args.rounds):
        try:
            logger.info(f"Starting round {round_num + 1}/{args.rounds}")
            success, metrics, history = client.federated_learning_round(
                X_train, y_train, X_test, y_test, epochs=args.epochs
            )

            if not success:
                logger.warning(f"Round {round_num + 1} failed")
                continue

            logger.info(f"Round {round_num + 1} completed successfully")

            # Log metrics
            loss = metrics['loss']
            accuracy = metrics['accuracy']
            auc = metrics['auc']
            logger.info(f"Round {round_num + 1} metrics - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

        except Exception as e:
            logger.error(f"Error in round {round_num + 1}: {e}")

    logger.info("Federated learning completed")


if __name__ == "__main__":
    main()