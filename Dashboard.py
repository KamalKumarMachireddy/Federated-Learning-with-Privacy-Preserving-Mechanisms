import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import datetime
import logging
import threading
import time
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MonitoringData:
    """Store and manage monitoring data for federated learning"""

    def __init__(self):
        self.rounds = []
        self.global_accuracy = []
        self.client_accuracy = {}
        self.privacy_budget = []
        self.update_times = []
        self.active_clients = []
        self.model_loss = []
        self.timestamps = []

    def add_round_data(self, round_num, global_acc, active_clients, privacy_spent):
        """Add data for a completed training round"""
        self.rounds.append(round_num)
        self.global_accuracy.append(global_acc)
        self.active_clients.append(active_clients)
        self.privacy_budget.append(privacy_spent)
        self.timestamps.append(datetime.datetime.now())

    def add_client_accuracy(self, round_num, client_id, accuracy):
        """Add client-specific accuracy data"""
        if client_id not in self.client_accuracy:
            self.client_accuracy[client_id] = {"rounds": [], "accuracy": []}

        self.client_accuracy[client_id]["rounds"].append(round_num)
        self.client_accuracy[client_id]["accuracy"].append(accuracy)

    def add_model_loss(self, round_num, loss):
        """Add global model loss data"""
        self.model_loss.append(loss)

    def get_global_metrics_df(self):
        """Get global metrics as a DataFrame"""
        return pd.DataFrame({
            "round": self.rounds,
            "accuracy": self.global_accuracy,
            "active_clients": self.active_clients,
            "privacy_budget": self.privacy_budget,
            "timestamp": self.timestamps
        })

    def get_client_accuracy_df(self):
        """Get client accuracy data as a DataFrame"""
        data = []
        for client_id, metrics in self.client_accuracy.items():
            for i in range(len(metrics["rounds"])):
                data.append({
                    "client_id": client_id,
                    "round": metrics["rounds"][i],
                    "accuracy": metrics["accuracy"][i]
                })
        return pd.DataFrame(data)


class FederatedLearningDashboard:
    """Dashboard for visualizing federated learning progress"""

    def __init__(self, monitoring_data=None):
        self.data = monitoring_data or MonitoringData()
        self.app = dash.Dash(__name__, title="Federated Learning Dashboard")
        self.setup_layout()
        self.setup_callbacks()
        logger.info("Dashboard initialized")

    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Federated Learning with Privacy - Monitoring Dashboard",
                    style={"textAlign": "center", "marginBottom": 30}),

            # System status card
            html.Div([
                html.H3("System Status", style={"marginBottom": 15}),
                html.Div([
                    html.Div([
                        html.H4("Current Round"),
                        html.Div(id="current-round", children="0")
                    ], className="status-card"),
                    html.Div([
                        html.H4("Active Clients"),
                        html.Div(id="active-clients", children="0")
                    ], className="status-card"),
                    html.Div([
                        html.H4("Model Accuracy"),
                        html.Div(id="model-accuracy", children="0%")
                    ], className="status-card"),
                    html.Div([
                        html.H4("Privacy Budget"),
                        html.Div(id="privacy-budget", children="0%")
                    ], className="status-card")
                ], style={"display": "flex", "justifyContent": "space-between"})
            ], style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "5px", "marginBottom": 20}),

            # Training progress
            html.Div([
                html.Div([
                    html.H3("Model Accuracy Over Training Rounds"),
                    dcc.Graph(id="accuracy-chart")
                ], style={"width": "48%"}),

                html.Div([
                    html.H3("Privacy Budget Consumption"),
                    dcc.Graph(id="privacy-chart")
                ], style={"width": "48%"})
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": 20}),

            # Client performance
            html.Div([
                html.H3("Client Performance Comparison"),
                dcc.Graph(id="client-comparison-chart")
            ], style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "5px", "marginBottom": 20}),

            # Refresh interval
            dcc.Interval(
                id="interval-component",
                interval=5 * 1000,  # in milliseconds (5 seconds)
                n_intervals=0
            )
        ])

    def setup_callbacks(self):
        """Setup the dashboard callbacks"""

        # Update system status cards
        @self.app.callback(
            [Output("current-round", "children"),
             Output("active-clients", "children"),
             Output("model-accuracy", "children"),
             Output("privacy-budget", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_status_cards(n):
            df = self.data.get_global_metrics_df()

            if df.empty:
                return "0", "0", "0%", "0%"

            latest = df.iloc[-1]
            current_round = str(latest["round"])
            active_clients = str(latest["active_clients"])
            model_accuracy = f"{latest['accuracy']:.2%}"
            privacy_budget = f"{latest['privacy_budget']:.2%}"

            return current_round, active_clients, model_accuracy, privacy_budget

        # Update accuracy chart
        @self.app.callback(
            Output("accuracy-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_accuracy_chart(n):
            df = self.data.get_global_metrics_df()

            if df.empty:
                # Return empty figure with message
                return go.Figure().add_annotation(
                    text="No data available", showarrow=False, font=dict(size=20)
                )

            fig = px.line(
                df, x="round", y="accuracy",
                markers=True,
                labels={"round": "Training Round", "accuracy": "Global Model Accuracy"},
                range_y=[0, 1]
            )

            fig.update_layout(
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
                yaxis=dict(showgrid=True, gridcolor="#e5e5e5", tickformat=".0%")
            )

            return fig

        # Update privacy budget chart
        @self.app.callback(
            Output("privacy-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_privacy_chart(n):
            df = self.data.get_global_metrics_df()

            if df.empty:
                # Return empty figure with message
                return go.Figure().add_annotation(
                    text="No data available", showarrow=False, font=dict(size=20)
                )

            # Create privacy budget figure
            fig = px.line(
                df, x="round", y="privacy_budget",
                markers=True,
                labels={"round": "Training Round", "privacy_budget": "Privacy Budget Spent"},
                range_y=[0, 1]
            )

            # Add threshold line at 100%
            fig.add_shape(
                type="line", line=dict(dash="dash", color="red", width=2),
                x0=0, x1=df["round"].max() + 1, y0=1, y1=1
            )

            fig.update_layout(
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
                yaxis=dict(showgrid=True, gridcolor="#e5e5e5", tickformat=".0%")
            )

            return fig

        # Update client comparison chart
        @self.app.callback(
            Output("client-comparison-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_client_chart(n):
            df = self.data.get_client_accuracy_df()

            if df.empty:
                # Return empty figure with message
                return go.Figure().add_annotation(
                    text="No client data available", showarrow=False, font=dict(size=20)
                )

            fig = px.line(
                df, x="round", y="accuracy", color="client_id",
                markers=True,
                labels={"round": "Training Round", "accuracy": "Local Model Accuracy", "client_id": "Client ID"},
                range_y=[0, 1]
            )

            fig.update_layout(
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
                yaxis=dict(showgrid=True, gridcolor="#e5e5e5", tickformat=".0%")
            )

            return fig

    def run_server(self, debug=False, port=8050):
        """Run the dashboard server"""
        logger.info(f"Starting dashboard server on port {port}")
        self.app.run(debug=debug, port=port)

    def simulate_data_generator(self, num_rounds=20, num_clients=3):
        """Generate simulated data for demonstration"""
        logger.info("Starting data simulation thread")

        def generate_data():
            privacy_spent = 0
            for round_num in range(1, num_rounds + 1):
                # Simulate global accuracy (improving over time)
                base_acc = 0.5 + 0.3 * (1 - np.exp(-0.15 * round_num))
                noise = np.random.normal(0, 0.02)
                global_acc = min(max(base_acc + noise, 0), 1)

                # Simulate active clients (random between 1 and num_clients)
                active = np.random.randint(1, num_clients + 1)

                # Simulate privacy budget (increases each round)
                privacy_spent += np.random.uniform(0.03, 0.06)
                privacy_spent = min(privacy_spent, 1.0)

                # Add global data
                self.data.add_round_data(round_num, global_acc, active, privacy_spent)

                # Add client-specific data
                for client_id in range(1, num_clients + 1):
                    if np.random.random() < 0.8:  # 80% chance client participated
                        client_acc = global_acc * np.random.uniform(0.8, 1.1)
                        client_acc = min(max(client_acc, 0), 1)
                        self.data.add_client_accuracy(round_num, f"Client {client_id}", client_acc)

                # Sleep to simulate time between rounds
                time.sleep(3)

        thread = threading.Thread(target=generate_data)
        thread.daemon = True
        thread.start()


# Example usage
if __name__ == "__main__":
    # Create dashboard
    dashboard = FederatedLearningDashboard()

    # Start simulating data in the background
    dashboard.simulate_data_generator(num_rounds=20, num_clients=5)

    # Run the dashboard
    dashboard.run_server(debug=True)