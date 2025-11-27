import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import warnings
import numpy as np

# 设置中文显示（解决中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
                # 添加异常处理，防止空文本导致的错误
                if text.strip():  # 确保文本不为空
                    # 使用更宽松的参数
                    keywords = jieba.analyse.textrank(text, topK=5, withWeight=False, allowPOS=('n', 'v', 'a'))
                    if keywords:  # 确保有关键词被提取
                        textrank_data.append(" ".join(keywords))
                    else:
                        textrank_data.append("")
                else:
                    textrank_data.append("")
            except Exception as e:
                # 如果出现任何异常，使用空字符串代替
                # print(f"TextRank处理文本时出错: {e}，文本内容: '{text}'")
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

# 朴素贝叶斯模型训练和预测
nb_classifier = MultinomialNB()
nb_classifier.fit(train_features_selected, train_labels)
nb_predictions = nb_classifier.predict(test_features_selected)

# 获取预测概率（用于绘制P-R和ROC曲线）
# predict_proba返回格式：[类别0概率, 类别1概率]，取类别1的概率作为正类概率
nb_probabilities = nb_classifier.predict_proba(test_features_selected)[:, 1]

# 性能指标
precision = precision_score(test_labels, nb_predictions)
recall = recall_score(test_labels, nb_predictions)
f1 = f1_score(test_labels, nb_predictions)

print("\n分类性能指标：")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))


# -------------------------- 新增：绘制P-R曲线 --------------------------
def plot_pr_curve(y_true, y_prob):
    # 计算精确率、召回率和阈值
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_prob)

    # 绘制P-R曲线
    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, color='darkorange', lw=2, label='P-R曲线')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('精确率-召回率曲线 (P-R Curve)')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)

    # 保存图片（可选）
    plt.savefig('result/21_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------- 新增：绘制ROC曲线 --------------------------
def plot_roc_curve(y_true, y_prob):
    # 计算FPR、TPR和阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # 计算AUC值
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # 随机猜测线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)')
    plt.ylabel('真正例率 (True Positive Rate)')
    plt.title('受试者工作特征曲线 (ROC Curve)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # 保存图片（可选）
    plt.savefig('result/21_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


# 调用函数绘制曲线
print("\n绘制P-R曲线和ROC曲线...")
plot_pr_curve(test_labels, nb_probabilities)
plot_roc_curve(test_labels, nb_probabilities)