import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class NumPyMLP:
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, lr=0.01):
        # 权重初始化
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b3 = np.zeros((1, output_dim))
        self.lr = lr

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.X = X
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3

    def backward(self, y_true, y_pred):
        m = y_true.shape[0]
        # 链式法则求导
        dz3 = 2 * (y_pred - y_true) / m
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_deriv(self.z2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_deriv(self.z1)
        dW1 = np.dot(self.X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 梯度下降更新
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=3000):
        loss_history = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(y, y_pred)
            loss = np.mean((y - y_pred)**2)
            loss_history.append(loss)
            if (epoch + 1) % 100 == 0:
                mae = np.mean(np.abs(y - y_pred))
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f} - mae: {mae:.4f}")
        return loss_history

def main():
    # 加载数据
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    X_train, y_train = train_df['x'].values.reshape(-1, 1), train_df['y'].values.reshape(-1, 1)
    X_test, y_test = test_df['x'].values.reshape(-1, 1), test_df['y'].values.reshape(-1, 1)

    # 训练模型
    print("开始 NumPy 模型训练...")
    model = NumPyMLP(lr=0.01)
    loss_history = model.train(X_train, y_train, epochs=3000)
    predictions = model.forward(X_test)
    print(f"Numpy 训练结束")
    
    test_loss = np.mean((y_test - predictions)**2)
    test_mae = np.mean(np.abs(y_test - predictions))
    
    print("-" * 30)
    print(f"NumPy 测试集最终评估:")
    print(f"Loss (MSE): {test_loss:.4f}")
    print(f"MAE (平均绝对误差): {test_mae:.4f}")
    print("-" * 30)

    # 保存数据和可视化
    os.makedirs('results', exist_ok=True)
    np.savetxt('results/numpy_loss.csv', loss_history, delimiter=',')
    np.savetxt('results/numpy_preds.csv', predictions, delimiter=',')

    plt.figure(figsize=(12, 5))
    
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, color='orange', label='NumPy Loss')
    plt.title('NumPy Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # 拟合效果
    plt.subplot(1, 2, 2)
    plt.scatter(X_train, y_train, color='gray', alpha=0.3, s=10, label='Train Data')
    plt.plot(X_test, y_test, color='blue', label='True Function', linewidth=2)
    plt.plot(X_test, predictions, color='red', linestyle='--', label='NumPy Predict', linewidth=2)
    plt.title('NumPy Function Fitting')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/numpy_results.png', dpi=300)

    train_predictions = model.forward(X_train)
    # 经验残差及其标准差 (Sigma)
    residuals = y_train - train_predictions
    std_dev = np.std(residuals)
    # 95% 统计置信区间
    upper_bound = predictions + 1.96 * std_dev
    lower_bound = predictions - 1.96 * std_dev
    plt.figure(figsize=(10, 6), dpi=300)
    plt.fill_between(
        X_test.flatten(), 
        lower_bound.flatten(), 
        upper_bound.flatten(), 
        color='dodgerblue', 
        alpha=0.2, 
        label='95% Confidence Interval ($\pm 1.96\sigma$)'
    )
    plt.scatter(
        X_train, y_train, 
        color='dimgray', 
        alpha=0.4, 
        s=10, 
        label='Raw Sampled Scatter'
    )
    plt.plot(
        X_test, y_test, 
        color='navy', 
        linewidth=2.5, 
        label='Objective True Function'
    )
    plt.plot(
        X_test, predictions, 
        color='crimson', 
        linestyle='--', 
        linewidth=2.5, 
        label='Predicted Trajectory'
    )
    plt.title('Neural Network Function Fitting with 95% Statistical Confidence Interval', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Input Variable ($X$)', fontsize=12)
    plt.ylabel('Target Variable ($Y$)', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/confidence_interval_fitting.png', bbox_inches='tight')
    
    print("结果和图表已保存至 results/numpy_results.png")

if __name__ == "__main__":
    main()