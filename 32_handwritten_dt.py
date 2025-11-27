import jieba
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score, f1_score
# 新增曲线相关导入
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import warnings

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
        seg_list = [word for word in jieba.cut(text) if word not in stopwords and word.strip()]
        seg_data.append(" ".join(seg_list))

        if use_textrank:
            try:
                if text.strip():  # 确保文本不为空
                    keywords = jieba.analyse.textrank(text, topK=5, withWeight=False, allowPOS=('n', 'v', 'a'))
                    if keywords:  # 确保有关键词被提取
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


def split_gini(X, y, feature_index, threshold):
    left_y, right_y = [], []
    for i, sample in enumerate(X):
        if sample[feature_index] < threshold:
            left_y.append(y[i])
        else:
            right_y.append(y[i])

    gini_left = gini_impurity(left_y) if left_y else 0
    gini_right = gini_impurity(right_y) if right_y else 0
    n_left, n_right = len(left_y), len(right_y)
    n_total = n_left + n_right

    if n_total == 0:  # 防止除以零
        return float("inf")

    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    return weighted_gini


class DecisionTreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class, all_classes):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class  # 各类别样本数（按all_classes顺序）
        self.predicted_class = predicted_class
        self.all_classes = all_classes  # 所有可能的类别（全局统一顺序）
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None


def gini_impurity(y):
    if len(y) == 0:
        return 0
    unique_classes, class_counts = np.unique(y, return_counts=True)
    impurity = 1.0
    total_samples = len(y)
    for count in class_counts:
        p_cls = count / total_samples
        impurity -= p_cls ** 2
    return impurity


def find_best_split(X, y, n_features):
    best_feature_index, best_threshold, best_gini = None, None, float("inf")

    if len(X) == 0:
        return best_feature_index, best_threshold

    for feature_index in range(n_features):
        thresholds = np.unique([row[feature_index] for row in X])
        for threshold in thresholds:
            gini = split_gini(X, y, feature_index, threshold)
            if gini < best_gini:
                best_gini, best_feature_index, best_threshold = gini, feature_index, threshold
    return best_feature_index, best_threshold


def split_dataset(X, y, feature_index, threshold):
    left_X, right_X, left_y, right_y = [], [], [], []
    for sample, label in zip(X, y):
        if sample[feature_index] < threshold:
            left_X.append(sample)
            left_y.append(label)
        else:
            right_X.append(sample)
            right_y.append(label)
    return left_X, left_y, right_X, right_y


def build_tree(X, y, all_classes, depth=0, max_depth=10, min_samples_split=5):
    """新增all_classes参数，确保全局类别顺序一致"""
    if len(X) == 0:
        # 空节点返回默认概率（均匀分布）
        num_samples_per_class = [0] * len(all_classes)
        predicted_class = all_classes[0] if all_classes else 0
        return DecisionTreeNode(
            gini=1.0,
            num_samples=0,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            all_classes=all_classes
        )

    # 统计各类别样本数（按all_classes顺序）
    num_samples_per_class = [0] * len(all_classes)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique_classes, class_counts):
        if cls in all_classes:
            idx = all_classes.index(cls)
            num_samples_per_class[idx] = count

    predicted_class = all_classes[np.argmax(num_samples_per_class)] if all_classes else 0

    node = DecisionTreeNode(
        gini=gini_impurity(y),
        num_samples=len(y),
        num_samples_per_class=num_samples_per_class,
        predicted_class=predicted_class,
        all_classes=all_classes
    )

    # 停止条件：达到最大深度、样本数不足、纯节点
    if (depth >= max_depth or
            len(X) < min_samples_split or
            len(np.unique(y)) == 1):
        return node

    n_features = len(X[0]) if len(X) > 0 else 0
    feature_index, threshold = find_best_split(X, y, n_features)

    if feature_index is not None:
        left_X, left_y, right_X, right_y = split_dataset(X, y, feature_index, threshold)

        if len(left_X) > 0 and len(right_X) > 0:
            node.feature_index = feature_index
            node.threshold = threshold
            node.left = build_tree(left_X, left_y, all_classes, depth + 1, max_depth, min_samples_split)
            node.right = build_tree(right_X, right_y, all_classes, depth + 1, max_depth, min_samples_split)

    return node


def predict(sample, node):
    if node is None:
        return 0  # 默认类别

    # 叶子节点返回预测类别
    if node.left is None and node.right is None:
        return node.predicted_class

    # 非叶子节点递归遍历
    if node.feature_index is not None and node.threshold is not None:
        if sample[node.feature_index] < node.threshold:
            return predict(sample, node.left)
        else:
            return predict(sample, node.right)

    return node.predicted_class


def predict_proba_single(sample, node):
    """新增：预测单个样本的类别概率（按all_classes顺序）"""
    if node is None:
        # 空节点返回均匀分布概率
        n_classes = len(node.all_classes) if node else 1
        return np.ones(n_classes) / n_classes

    # 叶子节点：返回各类别样本占比（概率）
    if node.left is None and node.right is None:
        total = node.num_samples
        if total == 0:
            # 空叶子节点返回均匀分布
            return np.ones(len(node.all_classes)) / len(node.all_classes)
        return np.array(node.num_samples_per_class) / total

    # 非叶子节点递归遍历
    if node.feature_index is not None and node.threshold is not None:
        if sample[node.feature_index] < node.threshold:
            return predict_proba_single(sample, node.left)
        else:
            return predict_proba_single(sample, node.right)

    # 无法分割时返回当前节点的类别分布
    total = node.num_samples
    if total == 0:
        return np.ones(len(node.all_classes)) / len(node.all_classes)
    return np.array(node.num_samples_per_class) / total


def predict_proba(samples, node):
    """新增：批量预测样本的类别概率（输出格式：n_samples × n_classes）"""
    if node is None or not node.all_classes:
        return np.zeros((len(samples), 1)) if len(samples) > 0 else np.array([])

    probabilities = []
    for sample in samples:
        prob = predict_proba_single(sample, node)
        probabilities.append(prob)
    return np.array(probabilities)


# 结合TF-IDF特征和TextRank特征
combined_train_features = [seg + " " + textrank for seg, textrank in zip(train_seg, train_textrank)]
combined_test_features = [seg + " " + textrank for seg, textrank in zip(test_seg, test_textrank)]

# 特征选择
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(combined_train_features)
test_features = vectorizer.transform(combined_test_features)

# 调整k值，确保不超过特征数量
k = min(40, train_features.shape[1])
selector = SelectKBest(chi2, k=k)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 打印选中的特征词
feature_names = vectorizer.get_feature_names_out()
selected_indices = selector.get_support(indices=True)
feature_words = [feature_names[idx] for idx in selected_indices]
print("特征词：", feature_words)

# 获取所有唯一类别（确保训练/测试类别一致）
all_classes = list(np.unique(train_labels))
print(f"\n数据类别：{all_classes}")

# 构建决策树（传入all_classes确保类别顺序一致）
print("开始构建决策树...")
tree = build_tree(
    train_features_selected.toarray(),
    train_labels,
    all_classes=all_classes,
    max_depth=22,
    min_samples_split=15
)

# 进行预测
print("开始预测...")
predictions = []
for instance in test_features_selected.toarray():
    pred = predict(instance, tree)
    predictions.append(pred)

# 获取正类概率（用于曲线绘制，仅适用于二元分类）
positive_probas = None
try:
    # 批量获取类别概率
    probas = predict_proba(test_features_selected.toarray(), tree)
    print(f"概率输出形状：{probas.shape}（样本数 × 类别数）")

    if len(all_classes) == 2:
        # 二元分类：取类别1的概率作为正类概率
        positive_cls = 1
        if positive_cls in all_classes:
            positive_idx = all_classes.index(positive_cls)
            positive_probas = probas[:, positive_idx]
            print("成功获取正类预测概率，将绘制曲线")
        else:
            print(f"警告：类别1不在数据类别中（数据类别：{all_classes}）")
    else:
        print(f"警告：当前为{len(all_classes)}分类任务，仅支持二元分类的曲线绘制")
except Exception as e:
    print(f"\n获取预测概率时出错: {e}，无法绘制曲线")

# 性能指标（保留4位小数）
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

print("\n自编写决策树模型性能:")
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
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, color='#8E44AD', lw=2.5, label='自编写决策树模型')
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('精确率-召回率曲线 (P-R Curve)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('result/32_pr_curve.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()


# -------------------------- 绘制ROC曲线（单模型） --------------------------
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='#1ABC9C', lw=2.5, label=f'自编写决策树模型 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#95A5A6', lw=1.5, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
    plt.title('受试者工作特征曲线 (ROC Curve)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('result/32_roc_curve.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()


# -------------------------- 调用函数绘制曲线 --------------------------
if positive_probas is not None:
    print("\n开始绘制P-R曲线和ROC曲线...")
    plot_pr_curve(test_labels, positive_probas)
    plot_roc_curve(test_labels, positive_probas)
else:
    print("\n跳过曲线绘制（不满足绘制条件）")