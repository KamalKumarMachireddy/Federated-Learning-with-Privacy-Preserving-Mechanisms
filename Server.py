import numpy as np
import grpc
import tensorflow as tf
from concurrent import futures
import time
import logging
from typing import List, Dict, Any
import pickle
import os
import federated_learning_pb2
import federated_learning_pb2_grpc

# Import the generated gRPC code (we'll define this later)
# import federated_pb2
# import federated_pb2_grpc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaillierEncryption:
    """
    Simple implementation of Paillier homomorphic encryption.
    In a real implementation, you would use a library like python-paillier.
    """

    def __init__(self):
        # In a real implementation, you would generate keys here
        logger.info("Initializing Paillier encryption")
        self.public_key = None
        self.private_key = None

    def encrypt(self, data):
        """Simulate encrypting data"""
        # Replace with actual encryption in production
        logger.info("Encrypting data with Paillier")
        return data

    def decrypt(self, encrypted_data):
        """Simulate decrypting data"""
        # Replace with actual decryption in production
        logger.info("Decrypting data with Paillier")
        return encrypted_data

    def aggregate_encrypted(self, encrypted_updates):
        """Aggregate encrypted updates homomorphically"""
        # In a real implementation, this would perform homomorphic addition
        logger.info("Performing homomorphic aggregation of encrypted updates")
        # Simple summation as placeholder for actual homomorphic addition
        return sum(encrypted_updates)


class DifferentialPrivacy:
    """
    Implementation of differential privacy mechanisms.
    """

    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta  # Failure probability
        logger.info(f"Initializing Differential Privacy with epsilon={epsilon}, delta={delta}")

    def add_noise_to_gradients(self, gradients, sensitivity=1.0):
        """Add Gaussian noise to gradients for differential privacy"""
        logger.info("Adding differential privacy noise to gradients")

        # Calculate the noise scale based on privacy parameters
        noise_scale = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon

        # Add Gaussian noise to each gradient
        noisy_gradients = []
        for grad in gradients:
            if isinstance(grad, np.ndarray):
                noise = np.random.normal(0, noise_scale, grad.shape)
                noisy_gradients.append(grad + noise)
            else:
                # Handle non-numpy types if needed
                noisy_gradients.append(grad)

        return noisy_gradients

    def clip_gradients(self, gradients, clip_value=1.0):
        """Clip gradients to limit sensitivity"""
        logger.info(f"Clipping gradients with threshold {clip_value}")

        clipped_gradients = []
        for grad in gradients:
            if isinstance(grad, np.ndarray):
                grad_norm = np.linalg.norm(grad)
                if grad_norm > clip_value:
                    grad = grad * (clip_value / grad_norm)
                clipped_gradients.append(grad)
            else:
                clipped_gradients.append(grad)

        return clipped_gradients


class ModelManager:
    """
    Manages the global model and aggregation logic.
    """

    def __init__(self, model_path=None):
        self.global_model = self._create_or_load_model(model_path)
        logger.info("ModelManager initialized with global model")

    def _create_or_load_model(self, model_path):
        """Create a new model or load an existing one"""
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            return tf.keras.models.load_model(model_path)
        else:
            logger.info("Creating new fraud detection model")
            # Create a simple model for fraud detection
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),  # Adjust input shape based on dataset
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
            return model

    def get_model_weights(self):
        """Get the current global model weights"""
        return self.global_model.get_weights()

    def update_model(self, aggregated_weights):
        """Update the global model with aggregated weights"""
        logger.info("Updating global model with aggregated weights")
        self.global_model.set_weights(aggregated_weights)

    def save_model(self, model_path="./global_model"):
        """Save the current global model"""
        logger.info(f"Saving model to {model_path}")
        self.global_model.save(model_path)

    def evaluate_model(self, test_data, test_labels):
        """Evaluate the global model on test data"""
        logger.info("Evaluating global model")
        return self.global_model.evaluate(test_data, test_labels)


class FederatedAggregator:
    """
    Handles aggregation of model updates from clients.
    """

    def __init__(self):
        self.encryption = PaillierEncryption()
        self.dp = DifferentialPrivacy()
        logger.info("FederatedAggregator initialized")

    def federated_averaging(self, model_updates, client_weights=None):
        """
        Perform simple federated averaging with optional weighting by client data size.

        Args:
            model_updates: List of model weight updates from clients
            client_weights: Optional weights for each client (e.g., data size)

        Returns:
            Aggregated model weights
        """
        logger.info(f"Performing federated averaging with {len(model_updates)} client updates")

        if not model_updates:
            raise ValueError("No model updates to aggregate")

        # If no client weights provided, use equal weighting
        if client_weights is None:
            client_weights = [1.0 / len(model_updates)] * len(model_updates)
        else:
            # Normalize weights to sum to 1
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]

        # Initialize aggregated weights with zeros like the first client's update
        aggregated_weights = [np.zeros_like(w) for w in model_updates[0]]

        # Weighted average of each layer's weights across clients
        for client_idx, client_update in enumerate(model_updates):
            weight = client_weights[client_idx]
            for layer_idx, layer_weights in enumerate(client_update):
                aggregated_weights[layer_idx] += layer_weights * weight

        return aggregated_weights

    def secure_aggregation(self, encrypted_updates, client_weights=None):
        """
        Perform secure aggregation using homomorphic encryption.

        In a real implementation, this would leverage homomorphic properties
        to aggregate without decrypting individual updates.
        """
        logger.info(f"Performing secure aggregation with {len(encrypted_updates)} encrypted updates")

        # In a real system, we would perform homomorphic addition
        # For demonstration, we're simulating by decrypting, aggregating, and re-encrypting
        decrypted_updates = [self.encryption.decrypt(update) for update in encrypted_updates]
        aggregated_weights = self.federated_averaging(decrypted_updates, client_weights)

        return aggregated_weights

    def privacy_preserving_aggregation(self, model_updates, client_weights=None):
        """
        Aggregate model updates with differential privacy.

        1. Clip gradients to bound sensitivity
        2. Add noise to achieve differential privacy
        3. Aggregate using federated averaging
        """
        logger.info("Performing privacy-preserving aggregation")

        # First clip gradients from each client
        clipped_updates = [self.dp.clip_gradients(update) for update in model_updates]

        # Add noise to achieve differential privacy
        noisy_updates = [self.dp.add_noise_to_gradients(update) for update in clipped_updates]

        # Perform federated averaging on the noisy updates
        return self.federated_averaging(noisy_updates, client_weights)


# This would be implemented in a separate proto file and generated using protoc
# For now we'll define placeholder classes to illustrate the server structure
class FederatedLearningServicer:
    """
    gRPC servicer that handles client communications.
    In a real implementation, this would be generated from a proto file.
    """

    def __init__(self):
        self.model_manager = ModelManager()
        self.aggregator = FederatedAggregator()
        self.current_round = 0
        self.client_updates = []
        self.client_weights = []
        self.min_clients_per_round = 2
        logger.info("FederatedLearningServicer initialized")

    def GetGlobalModel(self, request, context):
        """Send the current global model to clients"""
        logger.info(f"Client {request.client_id} requested global model")

        # In a real implementation, this would serialize model weights
        model_weights = self.model_manager.get_model_weights()
        serialized_weights = pickle.dumps(model_weights)

        # Placeholder for the actual protobuf response
        return {"model_weights": serialized_weights, "round": self.current_round}

    def SubmitUpdate(self, request, context):
        """Process model updates from clients"""
        client_id = request.client_id
        logger.info(f"Received model update from client {client_id} for round {request.round}")

        # Skip updates from outdated rounds
        if request.round != self.current_round:
            logger.warning(f"Client {client_id} submitted update for outdated round {request.round}")
            return {"success": False, "message": "Outdated round"}

        # Deserialize client model update
        client_update = pickle.loads(request.model_update)
        self.client_updates.append(client_update)
        self.client_weights.append(request.num_samples)

        # Check if we have enough updates to proceed to the next round
        if len(self.client_updates) >= self.min_clients_per_round:
            logger.info(f"Round {self.current_round} complete with {len(self.client_updates)} updates")
            self._aggregate_and_update()

        return {"success": True}

    def _aggregate_and_update(self):
        """Aggregate client updates and update the global model"""
        logger.info("Aggregating client updates and updating global model")

        # Perform privacy-preserving aggregation
        aggregated_weights = self.aggregator.privacy_preserving_aggregation(
            self.client_updates, self.client_weights
        )

        # Update global model
        self.model_manager.update_model(aggregated_weights)

        # Save model checkpoint
        self.model_manager.save_model(f"./model_checkpoints/global_model_round_{self.current_round}")

        # Clear updates and prepare for next round
        self.client_updates = []
        self.client_weights = []
        self.current_round += 1
        logger.info(f"Advanced to round {self.current_round}")


def serve():
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FederatedLearningServicer()

    # In a real implementation, this would register the generated servicer
    # federated_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, server)

    # For demonstration only - in real code, use a proper address
    port = 50051
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"Server started on port {port}")

    try:
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        server.stop(0)
        logger.info("Server stopped")


if __name__ == '__main__':
    # Create directory for model checkpoints
    os.makedirs("./model_checkpoints", exist_ok=True)
    logger.info("Starting federated learning central coordinator")
    serve()