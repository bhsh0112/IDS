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
        ])

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    X_train_tensor = torch.tensor(X_train_preprocessed.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_preprocessed.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return NetworkDataset(X_train_tensor, y_train_tensor), NetworkDataset(X_test_tensor, y_test_tensor)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = x.unsqueeze(1)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out)

def train_model(train_loader, test_loader, input_size, hidden_size, num_layers, num_epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size, hidden_size, num_layers, 1).to(device)
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
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                test_progress.set_postfix(accuracy=100 * correct / total)
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        current_f1 = f1_score(all_labels, all_preds)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {total_loss/len(train_loader):.4f}, '
              f'Test Acc: {100*correct/total:.2f}%, '
              f'F1-Score: {current_f1:.4f}')
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), "lstm_best_model.pth")
            print(f"Best model saved with F1-Score: {best_f1:.4f}")
    
    return model

if __name__ == "__main__":
    train_path = "./data/UNSW_NB15/UNSW_NB15_training-set.csv"
    test_path = "./data/UNSW_NB15/UNSW_NB15_testing-set.csv"
    batch_size = 512
    hidden_size = 128
    num_layers = 2
    num_epochs = 200
    learning_rate = 0.001

    train_dataset, test_dataset = load_data(train_path, test_path)
    input_size = train_dataset.features.shape[1]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = train_model(
        train_loader, test_loader,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_epochs=num_epochs,
        lr=learning_rate
    )