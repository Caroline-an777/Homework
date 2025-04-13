import numpy as np
import pickle
import os
import urllib.request
import tarfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_cifar10():
    data_dir = "cifar-10-batches-py"
    if not os.path.exists(data_dir):
        download_cifar10()
    return load_cifar10_batches(data_dir)

def download_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    save_path = "cifar-10-python.tar.gz"
    
    if not os.path.exists(save_path):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, save_path)
    
    with tarfile.open(save_path, "r:gz") as tar:
        tar.extractall()
    print("Dataset extracted to cifar-10-batches-py")

def load_cifar10_batches(data_dir):
    # 加载训练数据
    X_train, y_train = [], []
    for i in range(1, 6):
        with open(f"{data_dir}/data_batch_{i}", 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            X_train.append(data['data'])
            y_train.extend(data['labels'])
    
    # 加载测试数据
    with open(f"{data_dir}/test_batch", 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        X_test = data['data']
        y_test = data['labels']
    
    # 转换为numpy数组
    X_train = np.vstack(X_train).astype(np.float32)
    y_train = np.array(y_train)
    X_test = np.array(X_test).astype(np.float32)
    y_test = np.array(y_test)
    
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test

def split_validation(X_train_full, y_train_full, val_ratio=0.1):
    return train_test_split(
        X_train_full, y_train_full,
        test_size=val_ratio,
        stratify=y_train_full,
        random_state=42
    )