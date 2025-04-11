import os
import sys
import logging
import argparse
import subprocess
import time
import threading
import json
import socket
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("federated_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FederatedLearningSystem:
    """
    Main system integration class for federated learning project.

    This class is responsible for:
    1. Starting the central server
    2. Starting the dashboard
    3. Configuring and launching client instances
    4. Monitoring system health
    """

    def __init__(self, config_file=None):
        self.config = self._load_config(config_file)
        self.server_process = None
        self.dashboard_process = None
        self.client_processes = {}
        logger.info("Federated learning system initialized")

    def _load_config(self, config_file):
        """Load system configuration from a JSON file"""
        default_config = {
            "server": {
                "host": "localhost",
                "port": 50051,
                "model_path": "./models/initial_model",
                "privacy": {
                    "epsilon": 1.0,
                    "delta": 1e-5,
                    "clip_norm": 1.0
                }
            },
            "dashboard": {
                "port": 8050
            },
            "clients": {
                "count": 3,
                "data_path": "./data/fraud_data.csv",
                "local_epochs": 2,
                "batch_size": 32,
                "data_sampling_rate": 0.5,
                "privacy": {
                    "epsilon": 1.0,
                    "delta": 1e-5,
                    "clip_norm": 1.0
                }
            }
        }
        if config_file and Path(config_file).exists():
            logger.info(f"Loading configuration from {config_file}")
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning("No configuration file provided. Using default configuration.")
            return default_config

    def start_server(self):
        """Start the central federated learning server"""
        logger.info("Starting central server...")
        try:
            # Command to start the server script
            server_script = "central_server.py"  # Replace with the actual server script name
            server_command = [
                sys.executable, server_script,
                "--host", self.config["server"]["host"],
                "--port", str(self.config["server"]["port"]),
                "--model_path", self.config["server"]["model_path"]
            ]
            self.server_process = subprocess.Popen(
                server_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Central server started with PID {self.server_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start central server: {e}")

    def start_dashboard(self):
        """Start the monitoring dashboard"""
        logger.info("Starting monitoring dashboard...")
        try:
            # Command to start the dashboard script
            dashboard_script = "dashboard.py"  # Replace with the actual dashboard script name
            dashboard_command = [
                sys.executable, dashboard_script,
                "--port", str(self.config["dashboard"]["port"])
            ]
            self.dashboard_process = subprocess.Popen(
                dashboard_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Dashboard started with PID {self.dashboard_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")

    def start_clients(self):
        """Start client instances"""
        logger.info(f"Starting {self.config['clients']['count']} client instances...")
        try:
            client_script = "client.py"  # Replace with the actual client script name
            for i in range(self.config["clients"]["count"]):
                client_id = f"client_{i + 1}"
                client_command = [
                    sys.executable, client_script,
                    "--client_id", client_id,
                    "--server", f"{self.config['server']['host']}:{self.config['server']['port']}",
                    "--data_path", self.config["clients"]["data_path"],
                    "--epochs", str(self.config["clients"]["local_epochs"]),
                    "--batch_size", str(self.config["clients"]["batch_size"]),
                    "--sampling_rate", str(self.config["clients"]["data_sampling_rate"])
                ]
                client_process = subprocess.Popen(
                    client_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.client_processes[client_id] = client_process
                logger.info(f"Started client {client_id} with PID {client_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start clients: {e}")

    def stop_server(self):
        """Stop the central server"""
        if self.server_process:
            logger.info("Stopping central server...")
            self.server_process.terminate()
            self.server_process.wait()
            logger.info("Central server stopped.")
        else:
            logger.warning("No central server process to stop.")

    def stop_dashboard(self):
        """Stop the monitoring dashboard"""
        if self.dashboard_process:
            logger.info("Stopping monitoring dashboard...")
            self.dashboard_process.terminate()
            self.dashboard_process.wait()
            logger.info("Dashboard stopped.")
        else:
            logger.warning("No dashboard process to stop.")

    def stop_clients(self):
        """Stop all client instances"""
        logger.info("Stopping client instances...")
        for client_id, client_process in self.client_processes.items():
            try:
                client_process.terminate()
                client_process.wait()
                logger.info(f"Stopped client {client_id}.")
            except Exception as e:
                logger.error(f"Failed to stop client {client_id}: {e}")
        self.client_processes.clear()

    def monitor_system_health(self):
        """Monitor the health of the system components"""
        logger.info("Monitoring system health...")
        while True:
            # Check server process
            if self.server_process and self.server_process.poll() is not None:
                logger.error("Central server process has terminated unexpectedly.")
                self.start_server()

            # Check dashboard process
            if self.dashboard_process and self.dashboard_process.poll() is not None:
                logger.error("Dashboard process has terminated unexpectedly.")
                self.start_dashboard()

            # Check client processes
            for client_id, client_process in list(self.client_processes.items()):
                if client_process.poll() is not None:
                    logger.error(f"Client {client_id} process has terminated unexpectedly.")
                    del self.client_processes[client_id]

            time.sleep(10)  # Check every 10 seconds

    def run(self):
        """Run the federated learning system"""
        try:
            logger.info("Starting federated learning system...")
            self.start_server()
            self.start_dashboard()
            self.start_clients()

            # Start a background thread to monitor system health
            monitor_thread = threading.Thread(target=self.monitor_system_health, daemon=True)
            monitor_thread.start()

            logger.info("Federated learning system is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down federated learning system...")
        finally:
            self.stop_clients()
            self.stop_dashboard()
            self.stop_server()
            logger.info("Federated learning system shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="Federated Learning System Integration")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()

    system = FederatedLearningSystem(config_file=args.config)
    system.run()

if __name__ == "__main__":
    main()