import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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

train_data, train_labels = load_data("train.txt")
test_data, test_labels = load_data("test.txt")

# 数据预处理
def preprocess_data(data, use_textrank=True):
    seg_data = []
    textrank_data = []
    for text in data:
        seg_list = [word for word in jieba.cut(text) if word not in stopwords]
        seg_data.append(" ".join(seg_list))
        if use_textrank:
            keywords = " ".join(jieba.analyse.textrank(text, topK=5, withWeight=False))
            textrank_data.append(keywords)
    return seg_data, textrank_data

train_seg, train_textrank = preprocess_data(train_data)
test_seg, test_textrank = preprocess_data(test_data)

# 合并 TF-IDF 和关键词特征
combined_train_features = [seg + " " + textrank for seg, textrank in zip(train_seg, train_textrank)]
combined_test_features = [seg + " " + textrank for seg, textrank in zip(test_seg, test_textrank)]

# 向量化
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(combined_train_features)
test_features = vectorizer.transform(combined_test_features)

# 特征选择
selector = SelectKBest(chi2, k=20)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 特征词输出
feature_words = [vectorizer.get_feature_names_out()[idx] for idx in selector.get_support(indices=True)]
print("特征词：", feature_words)

# 逻辑回归模型（L2 正则防止过拟合）
lr_classifier = LogisticRegression(max_iter=1000, penalty='l2', solver='liblinear')
lr_classifier.fit(train_features_selected, train_labels)

# 预测
lr_predictions = lr_classifier.predict(test_features_selected)
lr_proba = lr_classifier.predict_proba(test_features_selected)[:, 1]  # 用于 ROC / P-R 曲线

# 计算指标
precision = precision_score(test_labels, lr_predictions)
recall = recall_score(test_labels, lr_predictions)
f1 = f1_score(test_labels, lr_predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# ROC 曲线
fpr, tpr, _ = roc_curve(test_labels, lr_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# P-R 曲线
precisions, recalls, _ = precision_recall_curve(test_labels, lr_proba)
plt.figure()
plt.plot(recalls, precisions, label='Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
