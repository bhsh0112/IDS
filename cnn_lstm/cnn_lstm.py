import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class NetworkDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    cols_to_drop = ['id', 'attack_cat']
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label'].values.astype(np.float32)
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label'].values.astype(np.float32)

    categorical_cols = ['proto', 'service', 'state']
    for col in categorical_cols:
        X_train[col] = X_train[col].replace('-', 'unknown')
        X_test[col] = X_test[col].replace('-', 'unknown')

    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    X_train_tensor = torch.tensor(X_train_preprocessed.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_preprocessed.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return NetworkDataset(X_train_tensor, y_train_tensor), NetworkDataset(X_test_tensor, y_test_tensor)

class CNNLSTMModel(nn.Module):
    def __init__(self, input_features, cnn_out_channels, lstm_hidden_size, num_layers, output_size):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        batch_size = cnn_out.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(cnn_out.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(cnn_out.device)
        lstm_out, _ = self.lstm(cnn_out, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)

def train_model(train_loader, test_loader, input_features, cnn_out_channels, lstm_hidden_size, num_layers, num_epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMModel(input_features, cnn_out_channels, lstm_hidden_size, num_layers, 1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())

        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        test_progress = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for inputs, labels in test_progress:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy().flatten().tolist())
                all_labels.extend(labels.cpu().numpy().flatten().tolist())
                test_progress.set_postfix(accuracy=100 * correct / total)
        f1 = f1_score(all_labels, all_preds)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Test Acc: {100*correct/total:.2f}%, F1-Score: {f1:.4f}')
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "cnn_lstm_best_model.pth")
            print(f"Best model saved at epoch {epoch+1} with F1-Score: {f1:.4f}")
    return model

if __name__ == "__main__":
    train_path = "./data/UNSW_NB15/UNSW_NB15_training-set.csv"
    test_path = "./data/UNSW_NB15/UNSW_NB15_testing-set.csv"

    train_dataset, test_dataset = load_data(train_path, test_path)
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_features = train_dataset.features.shape[1]
    cnn_out_channels = 32
    lstm_hidden_size = 128
    num_layers = 2
    num_epochs = 2000
    learning_rate = 0.001

    model = train_model(
        train_loader, test_loader,
        input_features=input_features,
        cnn_out_channels=cnn_out_channels,
        lstm_hidden_size=lstm_hidden_size,
        num_layers=num_layers,
        num_epochs=num_epochs,
        lr=learning_rate
    )