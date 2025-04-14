import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from cnn_lstm import load_data, CNNLSTMModel

def main():
    train_path = "./data/UNSW_NB15/UNSW_NB15_training-set.csv"
    test_path = "./data/NSL_KDD/NSL_KDD_Test.csv"
    
    _, test_dataset = load_data(train_path, test_path)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    input_features = test_dataset.features.shape[1]
    cnn_out_channels = 32
    lstm_hidden_size = 128
    num_layers = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMModel(input_features, cnn_out_channels, lstm_hidden_size, num_layers, 1).to(device)

    model_path = "./cnn_lstm/cnn_lstm_best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            all_preds.extend(predicted.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()