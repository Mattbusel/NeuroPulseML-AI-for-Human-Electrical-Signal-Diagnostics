import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.fft import fft, ifft
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def acquire_signal(signal_type='EEG', duration=10, fs=1000):
    """Simulate signal acquisition from EEG/EMG/EKG sensors."""
    t = np.linspace(0, duration, int(duration * fs))
    if signal_type == 'EEG':
        signal = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.1, len(t))  # Example EEG signal
    elif signal_type == 'EMG':
        signal = np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.2, len(t))  # Example EMG signal
    elif signal_type == 'EKG':
        signal = np.sin(2 * np.pi * 1 * t) + np.random.normal(0, 0.3, len(t))  # Example EKG signal
    return t, signal


def preprocess_signal(signal, fs=1000):
    """Preprocess signal using FFT and denoising."""
    
    fft_signal = fft(signal)
    
    denoised_signal = np.where(np.abs(fft_signal) > np.percentile(np.abs(fft_signal), 90), fft_signal, 0)
   
    signal_denoised = np.real(ifft(denoised_signal))
    return signal_denoised


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder, num_layers=num_layers)
        self.fc = nn.Linear(input_size, 1)  # Regression for anomaly score

    def forward(self, x):
        x = self.transformer_encoder(x)
        return self.fc(x[-1])

def train_model(train_data):
    """Train the transformer model on preprocessed signal data."""

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)

 
    X_train = torch.tensor(train_data_scaled, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(np.zeros(len(train_data)), dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

  
    model = TransformerModel(input_size=1, num_heads=4, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    for epoch in range(100):
        model.train()
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    return model


def detect_anomalies(signal_data):
    """Detect anomalies in signal data using Isolation Forest."""
    iso_forest = IsolationForest(contamination=0.05)
    anomalies = iso_forest.fit_predict(signal_data)
    return anomalies


def visualize_signal(signal, signal_denoised):
    """Visualize original and denoised signals."""
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal, label='Original Signal')
    plt.title("Original Signal")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(signal_denoised, label='Denoised Signal', color='orange')
    plt.title("Denoised Signal")
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_signal_3d(t, signal):
    """Visualize signal in 3D using Plotly."""
    trace = go.Scatter3d(
        x=t, y=signal, z=np.zeros_like(signal),
        mode='markers', marker=dict(size=5, color=signal, colorscale='Viridis')
    )
    layout = go.Layout(title="3D Signal Visualization", scene=dict(xaxis_title='Time', yaxis_title='Signal', zaxis_title=''))
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

if __name__ == "__main__":
    t, signal = acquire_signal(signal_type='EEG')
    signal_denoised = preprocess_signal(signal)
    

    visualize_signal(signal, signal_denoised)
    

    model = train_model(np.array([signal])) 
    
  
    anomalies = detect_anomalies(signal.reshape(-1, 1))
    
    
    visualize_signal_3d(t, signal)

    print("Anomaly Detection Results: ", anomalies)
