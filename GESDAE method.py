import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pywt

# 1. 数据预处理
def preprocess_data(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, :-1].values  # 特征
    y = data.iloc[:, -1].values  # 标签

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# 2. 时域特征提取
def extract_time_domain_features(X):
    time_domain_features = []
    for x in X:
        features = [
            np.mean(x),                  # 均值
            np.var(x),                   # 方差
            np.sqrt(np.mean(np.square(x))), # 均方根
            np.max(x),                   # 最大值
            np.min(x),                   # 最小值
            np.ptp(x),                   # 峰峰值
        ]
        time_domain_features.append(features)
    return np.array(time_domain_features)

# 3. 频域特征提取
def extract_frequency_domain_features(X):
    frequency_domain_features = []
    for x in X:
        fft_values = np.fft.fft(x)
        features = [
            np.abs(fft_values[:len(x)//2]).mean(),  # 平均频率
            np.std(np.abs(fft_values[:len(x)//2])),  # 频率标准差
        ]
        frequency_domain_features.append(features)
    return np.array(frequency_domain_features)

# 4. 时频域特征提取（假设使用小波变换）
def extract_time_frequency_features(X):
    time_frequency_domain_features = []
    for x in X:
        coeffs = pywt.wavedec(x, 'db4', level=4)
        features = [np.mean(coeff) for coeff in coeffs]
        time_frequency_domain_features.append(features)
    return np.array(time_frequency_domain_features)

# 5. 构建邻接矩阵
def build_adj_matrix(X):
    adj_matrix = cosine_similarity(X)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    return adj_matrix

# 6. 图嵌入模型（深度自编码器 + 图嵌入）
class GraphEmbeddingModel(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(GraphEmbeddingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = sparse_dropout(x, p=0.5)  # 使用稀疏化Dropout正则化
        x = torch.relu(self.fc2(x))
        x = sparse_dropout(x, p=0.5)  # 使用稀疏化Dropout正则化
        x = self.fc3(x)
        return x

# 7. 稀疏化Dropout正则化
def sparse_dropout(x, p=0.5):
    sorted_indices = torch.argsort(x.abs(), descending=False)
    num_elements = x.numel()
    drop_count = int(num_elements * p)
    keep_indices = sorted_indices[drop_count:]
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[keep_indices] = 1
    return x * mask.float()

# 8. Laplacian正则化：Laplacian矩阵计算
def laplacian_regularization(adj_matrix, embedding):
    degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
    laplacian_matrix = degree_matrix - adj_matrix
    return torch.norm(torch.mm(laplacian_matrix, embedding), p=2)

# 9. 损失函数：包括稀疏性惩罚和Laplacian正则化
class SparseLoss(nn.Module):
    def __init__(self, original_loss_fn, sparsity_penalty_factor=0.01):
        super(SparseLoss, self).__init__()
        self.original_loss_fn = original_loss_fn
        self.sparsity_penalty_factor = sparsity_penalty_factor

    def forward(self, x, target, adj_matrix=None, embedding=None):
        reconstruction_loss = self.original_loss_fn(x, target)
        sparsity_loss = self.sparsity_penalty_factor * torch.sum(torch.abs(x))
        laplacian_loss = laplacian_regularization(adj_matrix, embedding) if adj_matrix is not None else 0
        total_loss = reconstruction_loss + sparsity_loss + laplacian_loss
        return total_loss

# 10. 训练自编码器
def train_autoencoder(X_train, adj_matrix, embed_dim=64, lr=0.01, epochs=50, batch_size=128):
    model = GraphEmbeddingModel(input_dim=X_train.shape[1], embed_dim=5)  # 设置嵌入维度为5
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = SparseLoss(original_loss_fn=nn.MSELoss(), sparsity_penalty_factor=0.01)

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            batch_data = torch.tensor(X_train[i:i + batch_size], dtype=torch.float32)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_data, adj_matrix=adj_matrix, embedding=output)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model

# 11. KNN分类
def knn_classification(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Classification Accuracy: {accuracy * 100:.2f}%")

# 12. 超参数调优与交叉验证
def hyperparameter_tuning(X_train, y_train):
    param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree']}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_params_

# 主程序
def main():
    X, y = preprocess_data("data.xlsx")
    time_domain_features = extract_time_domain_features(X)
    frequency_domain_features = extract_frequency_domain_features(X)
    time_frequency_domain_features = extract_time_frequency_features(X)

    adj_matrix = build_adj_matrix(X)

    # 数据划分：150组训练，50组测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 训练模型
    model = train_autoencoder(X_train, adj_matrix, embed_dim=64, lr=0.01, epochs=50, batch_size=128)

    # KNN分类
    knn_classification(X_train, y_train, X_test, y_test)

    # 超参数调优
    best_params = hyperparameter_tuning(X_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Optimized KNN Classification Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
