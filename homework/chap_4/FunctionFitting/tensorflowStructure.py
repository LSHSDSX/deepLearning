import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Input 
import matplotlib.pyplot as plt
import os

def main():
    # 加载数据
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    X_train, y_train = train_df['x'].values.reshape(-1, 1), train_df['y'].values.reshape(-1, 1)
    X_test, y_test = test_df['x'].values.reshape(-1, 1), test_df['y'].values.reshape(-1, 1)

    # 构建与训练模型
    model = models.Sequential([
        Input(shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("开始 TensorFlow 模型训练...")
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    print(f"TensorFlow 训练结束")

    # 预测与保存数据
    tf_predictions = model.predict(X_test)

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print("-" * 30)
    print(f"TensorFlow 测试集最终评估:")
    print(f"Loss (MSE): {test_loss:.4f}")
    print(f"MAE (平均绝对误差): {test_mae:.4f}")
    print("-" * 30)

    os.makedirs('results', exist_ok=True)
    np.savetxt('results/tf_loss.csv', history.history['loss'], delimiter=',')
    np.savetxt('results/tf_preds.csv', tf_predictions, delimiter=',')

    # 可视化并保存图表
    plt.figure(figsize=(12, 5))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], color='green', label='TF Loss')
    plt.title('TensorFlow Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # 拟合效果
    plt.subplot(1, 2, 2)
    plt.scatter(X_train, y_train, color='gray', alpha=0.3, s=10, label='Train Data')
    plt.plot(X_test, y_test, color='blue', label='True Function', linewidth=2)
    plt.plot(X_test, tf_predictions, color='red', linestyle='--', label='TF Predict', linewidth=2)
    plt.title('TensorFlow Function Fitting')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/tf_results.png', dpi=300)
    
    train_predictions = model.predict(X_train)
    # 经验残差及其标准差 (Sigma)
    residuals = y_train - train_predictions
    std_dev = np.std(residuals)
    # 95% 统计置信区间
    upper_bound = tf_predictions + 1.96 * std_dev
    lower_bound = tf_predictions - 1.96 * std_dev
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
        X_test, tf_predictions, 
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
    print("结果和图表已保存至 results/tf_results.png")

if __name__ == "__main__":
    main()