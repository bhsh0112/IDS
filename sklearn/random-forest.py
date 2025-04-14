import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

    return X_train_preprocessed, y_train, X_test_preprocessed, y_test

def train_and_evaluate(train_path, test_path, n_estimators=100, random_state=42):
    print("加载数据...")
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)

    print("训练随机森林模型...")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("模型预测...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'测试准确率: {acc*100:.2f}%')

    return clf

if __name__ == "__main__":
    train_path = "./data/UNSW_NB15/UNSW_NB15_training-set.csv"
    test_path = "./data/UNSW_NB15/UNSW_NB15_testing-set.csv"
    model = train_and_evaluate(train_path, test_path)