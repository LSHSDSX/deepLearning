import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def target_function(x):
    # 目标函数： y = sin(x) + 0.5 * x
    return np.sin(x) + 0.5 * x

def generate_and_save_data():
    np.random.seed(42)

    # 数据采集
    X_total = np.random.uniform(-5, 5, 1250).reshape(-1, 1)
    y_total = target_function(X_total)

    # 使用 train_test_split 进行严格的互斥切分
    X_train, X_test, y_train, y_test = train_test_split(
        X_total, y_total, test_size=0.2, random_state=42
    )

    sort_idx = np.argsort(X_test.flatten())
    X_test = X_test[sort_idx]
    y_test = y_test[sort_idx]

    # 创建目录并保存为 CSV
    os.makedirs('data', exist_ok=True)

    train_df = pd.DataFrame({'x': X_train.flatten(), 'y': y_train.flatten()})
    test_df = pd.DataFrame({'x': X_test.flatten(), 'y': y_test.flatten()})

    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print("数据集采集与切分完毕！")
    print(f"全量数据集: {len(X_total)} 条")
    print(f"训练集 (80%): {len(X_train)} 条 (已保存至 data/train.csv)")
    print(f"测试集 (20%): {len(X_test)} 条 (已保存至 data/test.csv)")

if __name__ == "__main__":
    generate_and_save_data()