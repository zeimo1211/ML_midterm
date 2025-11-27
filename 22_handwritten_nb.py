import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score, f1_score
# 曲线相关导入
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import warnings
import numpy as np

# 解决中文乱码和负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.rcParams['axes.unicode_minus'] = False

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


# 加载训练和测试数据
train_data, train_labels = load_data("train.txt")
test_data, test_labels = load_data("test.txt")


# 分词、去除停用词及提取TextRank关键词
def preprocess_data(data, use_textrank=True):
    seg_data = []
    textrank_data = []
    for text in data:
        seg_list = [word for word in jieba.cut(text) if word not in stopwords and word.strip()]
        seg_data.append(" ".join(seg_list))

        if use_textrank:
            try:
                if text.strip():
                    keywords = jieba.analyse.textrank(text, topK=5, withWeight=False, allowPOS=('n', 'v', 'a'))
                    if keywords:
                        textrank_data.append(" ".join(keywords))
                    else:
                        textrank_data.append("")
                else:
                    textrank_data.append("")
            except Exception as e:
                textrank_data.append("")
    return seg_data, textrank_data


train_seg, train_textrank = preprocess_data(train_data)
test_seg, test_textrank = preprocess_data(test_data)

# 结合TF-IDF特征和TextRank特征
combined_train_features = [seg + " " + textrank for seg, textrank in zip(train_seg, train_textrank)]
combined_test_features = [seg + " " + textrank for seg, textrank in zip(test_seg, test_textrank)]

# 特征选择
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(combined_train_features)
test_features = vectorizer.transform(combined_test_features)

# 调整k值
k = max(20, train_features.shape[1])
selector = SelectKBest(chi2, k=k)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 打印选中的特征词
feature_names = vectorizer.get_feature_names_out()
selected_indices = selector.get_support(indices=True)
feature_words = [feature_names[idx] for idx in selected_indices]
print("特征词：", feature_words)


# 手写朴素贝叶斯模型（含predict_proba方法）
class HandwrittenMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 平滑参数
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None
        self.feature_count_ = None
        self.class_count_ = None

    def fit(self, X, y):
        # 转换为稠密数组以便处理
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        # 计算每个类别的先验概率和特征条件概率
        m, n = X_dense.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.feature_count_ = np.zeros((n_classes, n))
        self.class_count_ = np.zeros(n_classes)

        for i, cls in enumerate(self.classes_):
            mask = (y == cls)
            X_cls = X_dense[mask]
            self.feature_count_[i, :] = X_cls.sum(axis=0)
            self.class_count_[i] = X_cls.shape[0]

        # 类先验概率（对数形式）
        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())

        # 特征条件概率（对数形式，拉普拉斯平滑）
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

    def predict(self, X):
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        # 计算每个类别的对数概率
        jll = X_dense.dot(self.feature_log_prob_.T) + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        """输出每个类别的概率（用于曲线绘制）"""
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X

        # 计算对数后验概率
        jll = X_dense.dot(self.feature_log_prob_.T) + self.class_log_prior_
        # 转换为普通概率（使用softmax保证数值稳定）
        exp_jll = np.exp(jll - np.max(jll, axis=1, keepdims=True))
        proba = exp_jll / exp_jll.sum(axis=1, keepdims=True)
        return proba


# -------------------------- 模型训练与预测 --------------------------
print("\n训练手写朴素贝叶斯模型...")
handwritten_nb = HandwrittenMultinomialNB(alpha=1.0)
handwritten_nb.fit(train_features_selected, train_labels)
handwritten_preds = handwritten_nb.predict(test_features_selected)
handwritten_probs = handwritten_nb.predict_proba(test_features_selected)[:, 1]  # 正类（类别1）概率

# -------------------------- 性能指标计算 --------------------------
handwritten_precision = precision_score(test_labels, handwritten_preds)
handwritten_recall = recall_score(test_labels, handwritten_preds)
handwritten_f1 = f1_score(test_labels, handwritten_preds)

# 打印性能指标
print("\n手写朴素贝叶斯模型性能:")
print(f"Precision: {handwritten_precision:.4f}")
print(f"Recall:    {handwritten_recall:.4f}")
print(f"F1-score:  {handwritten_f1:.4f}")


# -------------------------- 绘制P-R曲线（单模型） --------------------------
def plot_pr_curve(y_true, y_prob):
    # 计算精确率、召回率
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, color='darkorange', lw=2, label='手写朴素贝叶斯')
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('精确率-召回率曲线 (P-R Curve)', fontsize=14)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('result/22_pr_curve.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()


# -------------------------- 绘制ROC曲线（单模型） --------------------------
def plot_roc_curve(y_true, y_prob):
    # 计算FPR、TPR和AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'手写朴素贝叶斯 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机猜测')  # 基准线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
    plt.title('受试者工作特征曲线 (ROC Curve)', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig('result/22_roc_curve.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()


# -------------------------- 调用函数绘制曲线 --------------------------
print("\n绘制P-R曲线和ROC曲线...")
plot_pr_curve(test_labels, handwritten_probs)
plot_roc_curve(test_labels, handwritten_probs)