import pandas as pd
import numpy as np
import math
import time
import os
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import amp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# 设置随机种子确保可复现性
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 忽略警告
warnings.filterwarnings('ignore')

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class NetworkDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """入侵检测Transformer模型"""
    def __init__(self, input_features, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, num_classes=1):
        super(TransformerModel, self).__init__()
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_features, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 输入形状: [batch_size, features]
        x = self.input_embedding(x)  # [batch_size, d_model]
        x = x.unsqueeze(1)  # 添加序列维度: [batch_size, seq_len=1, d_model]
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)  # 移除序列维度: [batch_size, d_model]
        return self.classifier(x)

def load_and_preprocess_data(train_path, test_path):
    """加载和预处理数据"""
    print("Loading and preprocessing data...")
    
    # 读取数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 打印数据集信息
    print(f"Training set size: {train_df.shape}")
    print(f"Test set size: {test_df.shape}")
    
    # 基本特征工程
    print("Performing feature engineering...")
    for df in [train_df, test_df]:
        # 添加新特征
        df['duration_per_packet'] = df['dur'] / (df['spkts'] + df['dpkts'] + 1e-5)
        df['bytes_per_packet'] = (df['sbytes'] + df['dbytes']) / (df['spkts'] + df['dpkts'] + 1e-5)
        df['packet_size_variance'] = df[['spkts', 'dpkts']].var(axis=1)
        df['packet_imbalance'] = (df['spkts'] - df['dpkts']) / (df['spkts'] + df['dpkts'] + 1e-5)
        df['tcp_flag_combination'] = df['stcpb'] | df['dtcpb']  # 组合TCP标志位
        
        # 处理无穷大值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
    
    # 删除不需要的列
    cols_to_drop = ['id', 'attack_cat']
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
    
    # 分离特征和标签
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label'].values.astype(np.float32)
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label'].values.astype(np.float32)
    
    # 处理类别不平衡
    print("Addressing class imbalance...")
    print(f"Class distribution before balancing: Normal={np.sum(y_train == 0)}, Attack={np.sum(y_train == 1)}")
    
    # 使用SMOTE过采样
    smote = SMOTE(sampling_strategy='minority', random_state=SEED)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"Class distribution after balancing: Normal={np.sum(y_train == 0)}, Attack={np.sum(y_train == 1)}")
    
    # 处理类别特征
    categorical_cols = ['proto', 'service', 'state']
    print("Processing categorical features...")
    for col in categorical_cols:
        # 合并低频类别
        train_counts = X_train[col].value_counts()
        low_freq_train = train_counts[train_counts < 100].index
        test_counts = X_test[col].value_counts()
        low_freq_test = test_counts[test_counts < 50].index
        
        # 替换特殊值和低频值
        X_train[col] = X_train[col].replace('-', 'unknown')
        X_test[col] = X_test[col].replace('-', 'unknown')
        X_train[col] = X_train[col].replace(list(low_freq_train), 'other')
        X_test[col] = X_test[col].replace(list(low_freq_test), 'other')
    
    # 数值特征
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
    
    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # 应用预处理
    print("Applying preprocessing...")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # 特征选择
    print("Performing feature selection...")
    selector = SelectKBest(f_classif, k=min(60, X_train_preprocessed.shape[1]))
    X_train_selected = selector.fit_transform(X_train_preprocessed, y_train)
    X_test_selected = selector.transform(X_test_preprocessed)
    
    print(f"Selected {X_train_selected.shape[1]} features out of {X_train_preprocessed.shape[1]}")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # 创建数据集
    train_dataset = NetworkDataset(X_train_tensor, y_train_tensor)
    test_dataset = NetworkDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset, y_train

def create_data_loaders(train_dataset, test_dataset, batch_size=1024):
    """创建数据加载器"""
    print("Creating data loaders...")
    
    # 创建训练集和验证集的划分
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, y_train, 
                num_epochs=100, lr=0.001, early_stopping_patience=15):
    """训练模型"""
    print("Starting model training...")
    
    # 将模型移至设备
    model = model.to(device)
    
    # 计算类别权重
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    pos_weight = torch.tensor([neg_count / (pos_count + 1e-5)]).to(device)
    print(f"Positive weight: {pos_weight.item():.2f}")
    
    # 损失函数 - 带类别权重
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True,
        min_lr=1e-6
    )
    
    # 混合精度训练
    scaler = amp.GradScaler()
    
    # 训练跟踪变量
    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    history = {
        'train_loss': [], 
        'val_loss': [],
        'val_f1': [], 
        'val_recall': [],
        'val_precision': []
    }
    
    # 训练循环
    start_time = time.time()
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新权重
            scaler.step(optimizer)
            scaler.update()
            
            # 更新进度条
            epoch_train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
        
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                all_preds.extend(preds.cpu().numpy().flatten().tolist())
                all_labels.extend(labels.cpu().numpy().flatten().tolist())
                all_probs.extend(probs.cpu().numpy().flatten().tolist())
        
        # 计算验证指标
        avg_val_loss = val_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        
        # 记录历史
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(f1)
        history['val_recall'].append(recall)
        history['val_precision'].append(precision)
        
        # 打印指标
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # 更新学习率
        scheduler.step(f1)
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            no_improve = 0
            
            # 保存完整模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'f1': f1,
                'recall': recall,
                'precision': precision,
                'history': history
            }, "best_transformer_model.pth")
            
            print(f"Best model saved with F1: {f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"No improvement for {early_stopping_patience} epochs. Early stopping...")
                break
    
    # 训练完成
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes. Best F1: {best_f1:.4f} at epoch {best_epoch}")
    
    # 加载最佳模型
    checkpoint = torch.load("best_transformer_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

def evaluate_model(model, test_loader):
    """评估模型性能"""
    print("Evaluating model on test set...")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
    
    # 计算评估指标
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # 分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Attack']))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 返回评估结果
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'probs': all_probs,
        'preds': all_preds,
        'labels': all_labels
    }

def plot_training_history(history):
    """绘制训练历史图表"""
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # F1分数曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['val_f1'], label='Validation F1', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.grid(True)
    
    # 精确率和召回率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['val_precision'], label='Precision', color='blue')
    plt.plot(history['val_recall'], label='Recall', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Precision and Recall')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def find_optimal_threshold(model, val_loader):
    """寻找最佳分类阈值"""
    print("Finding optimal classification threshold...")
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            
            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
    
    # 尝试不同的阈值
    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        preds = (np.array(all_probs) >= thresh).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    print(f"Optimal threshold found: {best_thresh:.4f} (F1={best_f1:.4f})")
    return best_thresh

if __name__ == "__main__":
    # 数据集路径
    train_path = "./data/UNSW_NB15/UNSW_NB15_training-set.csv"
    test_path = "./data/UNSW_NB15/UNSW_NB15_testing-set.csv"
    
    # 加载和预处理数据
    train_dataset, test_dataset, y_train = load_and_preprocess_data(train_path, test_path)
    
    # 创建数据加载器
    batch_size = 2048  # 更大的批大小可以更好地利用GPU
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, test_dataset, batch_size)
    
    # 模型参数
    input_features = train_dataset.features.shape[1]
    print(f"\nNumber of features: {input_features}")
    
    # 创建Transformer模型
    model = TransformerModel(
        input_features=input_features,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.2
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(model)
    
    # 训练模型
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_train=y_train,
        num_epochs=100,
        lr=0.0005,
        early_stopping_patience=15
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 寻找最佳阈值
    optimal_threshold = find_optimal_threshold(trained_model, val_loader)
    
    # 评估模型
    eval_results = evaluate_model(trained_model, test_loader)
    
    # 保存最终模型
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'input_features': input_features,
        'eval_results': eval_results
    }, "final_transformer_model.pth")
    print("Final model saved.")