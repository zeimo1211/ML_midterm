import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
# 新增曲线相关导入
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
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

# 检查数据是否加载成功
if not train_data or not test_data:
    print("错误：训练或测试数据加载失败，程序退出")
    exit()


# 分词、去除停用词及提取TextRank关键词
def preprocess_data(data, use_textrank=True):
    seg_data = []
    textrank_data = []
    for text in data:
        # 分词并去除停用词
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

# 调整k值，确保不超过特征数量
k = max(40, train_features.shape[1])
selector = SelectKBest(chi2, k=k)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 打印选中的特征词
feature_names = vectorizer.get_feature_names_out()
selected_indices = selector.get_support(indices=True)
feature_words = [feature_names[idx] for idx in selected_indices]
# print("特征词：", feature_words)

# 决策树模型训练和预测
clf = DecisionTreeClassifier()
clf.fit(train_features_selected, train_labels)
predictions = clf.predict(test_features_selected)

# 获取正类概率（用于曲线绘制，仅适用于二元分类）
try:
    # 决策树支持predict_proba，输出格式为(n_samples, n_classes)
    probas = clf.predict_proba(test_features_selected)
    if probas.shape[1] == 2:
        positive_probas = probas[:, 1]  # 取类别1的概率作为正类概率
        print("\n成功获取正类预测概率，将绘制曲线")
    else:
        positive_probas = None
        print(f"\n警告：当前为{probas.shape[1]}分类任务，仅支持二元分类的曲线绘制")
except Exception as e:
    positive_probas = None
    print(f"\n获取预测概率时出错: {e}，无法绘制曲线")

# 性能指标计算（保留4位小数，更美观）
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

print("\n决策树模型性能:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# 计算并打印AUC（仅二元分类）
if positive_probas is not None:
    try:
        auc_score = roc_auc_score(test_labels, positive_probas)
        print(f"AUC:       {auc_score:.4f}")
    except Exception as e:
        print(f"计算AUC时出错: {e}")


# -------------------------- 绘制P-R曲线（单模型） --------------------------
def plot_pr_curve(y_true, y_prob):
    # 计算精确率、召回率
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, color='#2ECC71', lw=2.5, label='决策树模型')
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('精确率-召回率曲线 (P-R Curve)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()  # 自动调整布局，避免标签截断
    plt.savefig('result/31_pr_curve.png', dpi=300, bbox_inches='tight')  # 保存高清图片
    plt.show()


# -------------------------- 绘制ROC曲线（单模型） --------------------------
def plot_roc_curve(y_true, y_prob):
    # 计算FPR、TPR和AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    # 模型ROC曲线
    plt.plot(fpr, tpr, color='#F39C12', lw=2.5, label=f'决策树模型 (AUC = {roc_auc:.4f})')
    # 随机猜测基准线
    plt.plot([0, 1], [0, 1], color='#95A5A6', lw=1.5, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
    plt.title('受试者工作特征曲线 (ROC Curve)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('result/31_roc_curve.png', dpi=300, bbox_inches='tight')  # 保存高清图片
    plt.show()


# -------------------------- 调用函数绘制曲线（仅当条件满足时） --------------------------
if positive_probas is not None:
    print("\n开始绘制P-R曲线和ROC曲线...")
    plot_pr_curve(test_labels, positive_probas)
    plot_roc_curve(test_labels, positive_probas)
else:
    print("\n跳过曲线绘制（不满足绘制条件）")