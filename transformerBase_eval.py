from matplotlib import pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch
import seaborn as sns
from imblearn.over_sampling import SMOTE

from transformerBase_train import NetworkDataset, TransformerModel

# # 设置随机种子确保可复现性
# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_model(val_dataset, model_path):
    """加载训练好的模型"""
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型实例
    input_features = val_dataset.features.shape[1]
    model = TransformerModel(
        input_features=input_features,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.2
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # 设置为评估模式
    
    print(f"Model loaded from {model_path}")
    
    return model

def load_and_preprocess_data(val_path):
    """加载和预处理数据"""
    print("Loading and preprocessing data...")
    
    # 读取数据
    val_df= pd.read_csv(val_path)
    
    # 打印数据集信息
    print(f"Training set size: {val_df.shape}")
    
    # 基本特征工程
    print("Performing feature engineering...")
    # 添加新特征
    val_df['duration_per_packet'] = val_df['dur'] / (val_df['spkts'] + val_df['dpkts'] + 1e-5)
    val_df['bytes_per_packet'] = (val_df['sbytes'] + val_df['dbytes']) / (val_df['spkts'] + val_df['dpkts'] + 1e-5)
    val_df['packet_size_variance'] = val_df[['spkts', 'dpkts']].var(axis=1)
    val_df['packet_imbalance'] = (val_df['spkts'] - val_df['dpkts']) / (val_df['spkts'] + val_df['dpkts'] + 1e-5)
    val_df['tcp_flag_combination'] = val_df['stcpb'] | val_df['dtcpb']  # 组合TCP标志位 
    # 处理无穷大值
    val_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    val_df.fillna(0, inplace=True)
    
    # 删除不需要的列
    cols_to_drop = ['id', 'attack_cat']
    val_df = val_df.drop(columns=cols_to_drop, errors='ignore')
    
    # 分离特征和标签
    X_val = val_df.drop(columns=['label'])
    y_val = val_df['label'].values.astype(np.float32)
    
    # 识别特征类型
    categorical_cols = ['proto', 'service', 'state']
    numeric_cols = [col for col in X_val.columns if col not in categorical_cols]
    
    # 创建预处理管道 - 统一处理数值和类别特征
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # 应用预处理
    print("Applying preprocessing...")
    X_val_preprocessed = preprocessor.fit_transform(X_val)
    
    # 处理类别不平衡（在预处理后执行！）
    # print("Addressing class imbalance...")
    # print(f"Class distribution before balancing: Normal={np.sum(y_val == 0)}, Attack={np.sum(y_val == 1)}")
    
    # # 使用SMOTE过采样
    # smote = SMOTE(sampling_strategy='minority', random_state=SEED)
    # X_val_preprocessed, y_val = smote.fit_resample(X_val_preprocessed, y_val)
    # print(f"Class distribution after balancing: Normal={np.sum(y_val == 0)}, Attack={np.sum(y_val == 1)}")
    
    # 特征选择
    print("Performing feature selection...")
    selector = SelectKBest(f_classif, k=min(60, X_val_preprocessed.shape[1]))
    X_val_selected = selector.fit_transform(X_val_preprocessed,y_val)
    
    print(f"Selected {X_val_selected.shape[1]} features out of {X_val_preprocessed.shape[1]}")
    
    # 转换为PyTorch张量
    X_val_tensor = torch.tensor(X_val_selected, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    # 创建数据集
    val_dataset = NetworkDataset(X_val_tensor, y_val_tensor)
    
    return val_dataset

def create_val_dataloader(val_dataset, batch_size):
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    return val_loader

def evaluate_model(model, val_loader):
    """评估模型性能"""
    print("Evaluating model on test set...")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
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

    #整体报告
    print("\nWhole Report:")
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Accuracy:", accuracy)

    
    # 分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Attack']))    
    
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

val_path = "./data/UNSW_NB15_training-set.csv"
model_path="./runs/best_transformer_model.pth"
val_dataset = load_and_preprocess_data(val_path)  
val_loader= create_val_dataloader(val_dataset, batch_size=512)
model=load_model(val_dataset,model_path)
eval_results = evaluate_model(model, val_loader)