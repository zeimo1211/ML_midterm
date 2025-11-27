#cell 1 通用数据预处理部分

import jieba
import jieba.analyse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

# 加载停用词
stopwords = set()
with open("stopwords.txt", "r", encoding="utf-8") as f:
    for line in f:
        stopwords.add(line.strip())

# 加载数据函数
def load_data(filepath):
    data, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                data.append(parts[2])
                labels.append(int(parts[1]))
    return data, labels

# 文本预处理：分词 + 去停用词 + TextRank关键词
def preprocess_data(data, use_textrank=True):
    seg_data = []
    textrank_data = []
    for text in data:
        seg_list = [word for word in jieba.cut(text) if word not in stopwords]
        seg_data.append(" ".join(seg_list))

        if use_textrank:
            keywords = " ".join(jieba.analyse.textrank(text, topK=5, withWeight=False))
            textrank_data.append(keywords)
        else:
            textrank_data.append("")
    return seg_data, textrank_data

# 读取训练与测试数据
train_data, train_labels = load_data("train.txt")
test_data, test_labels = load_data("test.txt")

# 预处理
train_seg, train_textrank = preprocess_data(train_data, use_textrank=True)
test_seg, test_textrank = preprocess_data(test_data, use_textrank=True)

# 合并 TF-IDF 与 TextRank 特征文本
combined_train_features = [seg + " " + textrank for seg, textrank in zip(train_seg, train_textrank)]
combined_test_features = [seg + " " + textrank for seg, textrank in zip(test_seg, test_textrank)]

# 向量化
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(combined_train_features)
test_features = vectorizer.transform(combined_test_features)

# 特征选择（卡方）
selector = SelectKBest(chi2, k=20)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 打印选中特征词
feature_words = [vectorizer.get_feature_names_out()[idx] for idx in selector.get_support(indices=True)]
print("选中特征词：", feature_words)

# =============================================================================
# Cell 2: 逻辑回归 (Scikit-learn 标准实现)
# =============================================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. 模型配置与训练
# -----------------------------------------------------------------------------
lr_model = LogisticRegression(
    max_iter=1000, 
    penalty='l2', 
    solver='liblinear', 
    random_state=42
)
lr_model.fit(train_features_selected, np.array(train_labels))

# 2. 模型推理
# -----------------------------------------------------------------------------
# 获取预测类别和正类概率
lr_pred = lr_model.predict(test_features_selected)
lr_proba = lr_model.predict_proba(test_features_selected)[:, 1]

# 3. 指标计算
# -----------------------------------------------------------------------------
precision = precision_score(test_labels, lr_pred)
recall = recall_score(test_labels, lr_pred)
f1 = f1_score(test_labels, lr_pred)

# 4. 结果输出
# -----------------------------------------------------------------------------
print("=" * 40)
print(f"【模型 1】Scikit-learn 逻辑回归")
print("-" * 40)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("=" * 40)
