import os
import jieba
import jieba.analyse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import time
import datetime

from common import load_data, load_stopwords, preprocess_data, extract_features


# 函数：格式化时间显示
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_random_forest():
    # 加载数据
    train_data, train_labels = load_data("train.txt")
    test_data, test_labels = load_data("test.txt")

    print(f"训练数据数量: {len(train_data)}")
    print(f"测试数据数量: {len(test_data)}")

    # 加载停用词
    stopwords = load_stopwords("stopwords.txt")

    # 预处理数据
    print("预处理数据...")
    train_seg, train_textrank = preprocess_data(train_data, stopwords, use_textrank=True)
    test_seg, test_textrank = preprocess_data(test_data, stopwords, use_textrank=True)

    # 提取特征
    print("提取特征...")
    train_features, test_features = extract_features(
        train_seg, train_textrank, test_seg, test_textrank, train_labels,
        vectorizer_params={'max_features': 5000}, k=20, print_features=True
    )

    # 训练随机森林模型
    print("训练随机森林模型...")
    t0 = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(train_features, train_labels)
    training_time = format_time(time.time() - t0)
    print(f"训练用时: {training_time}")

    # 确保保存目录存在
    os.makedirs('./saved_rf_model', exist_ok=True)

    # 保存模型和特征提取器
    with open('./saved_rf_model/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

    # 保存测试数据用于后续推理
    test_data_dict = {
        'test_features': test_features,
        'test_labels': test_labels
    }

    with open('./rf_test_data.pkl', 'wb') as f:
        pickle.dump(test_data_dict, f)

    print("随机森林模型训练完成并已保存")


if __name__ == "__main__":
    train_random_forest()