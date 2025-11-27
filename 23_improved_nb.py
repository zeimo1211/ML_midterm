import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# 新增曲线相关导入
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 解决中文乱码和负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.rcParams['axes.unicode_minus'] = False

## nb+cart
# 加载停用词
stopwords = set()
try:
    with open("stopwords.txt", "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:  # 确保不是空行
                stopwords.add(word)
except FileNotFoundError:
    print("警告：停用词文件未找到，将不使用停用词")
except Exception as e:
    print(f"加载停用词时出错: {e}")


# 加载数据函数
def load_data(filepath):
    data, labels = [], []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    text = parts[2].strip()
                    if text:  # 确保文本不为空
                        data.append(text)
                        labels.append(int(parts[1]))
    except FileNotFoundError:
        print(f"文件 {filepath} 未找到")
    except Exception as e:
        print(f"加载数据时出错: {e}")
    return data, labels


# 加载训练和测试数据
train_data, train_labels = load_data("train.txt")
test_data, test_labels = load_data("test.txt")

# 检查数据是否加载成功
if not train_data or not test_data:
    print("错误：未能加载训练或测试数据")
    exit()

print(f"训练数据数量: {len(train_data)}")
print(f"测试数据数量: {len(test_data)}")


# 分词和去除停用词
def segment_texts(texts):
    seg_texts = []
    for text in texts:
        try:
            seg_list = [word for word in jieba.cut(text) if word not in stopwords and word.strip()]
            seg_texts.append(" ".join(seg_list))
        except Exception as e:
            print(f"分词时出错: {e}，文本: {text}")
            seg_texts.append("")  # 添加空字符串作为占位符
    return seg_texts


train_seg = segment_texts(train_data)
test_seg = segment_texts(test_data)

# 特征提取
vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)  # 添加参数过滤罕见和常见词
train_features = vectorizer.fit_transform(train_seg)
test_features = vectorizer.transform(test_seg)

print(f"原始特征数量: {train_features.shape[1]}")

# 特征选择 - 安全地设置k值
k = min(15000, train_features.shape[1])
if k < 15000:
    print(f"警告：实际特征数量只有 {train_features.shape[1]}，将k调整为 {k}")

selector = SelectKBest(chi2, k=k)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 打印选中的特征词
feature_names = vectorizer.get_feature_names_out()
selected_indices = selector.get_support(indices=True)
if len(selected_indices) > 0:
    feature_words = [feature_names[idx] for idx in selected_indices[:10]]  # 只显示前10个
    print("前10个特征词：", feature_words)
else:
    print("警告：没有选择到任何特征词")
    # 如果不选择特征，使用原始特征
    train_features_selected = train_features
    test_features_selected = test_features

# 贝叶斯模型训练和预测
clf = MultinomialNB(alpha=1.0)  # 添加平滑参数
clf.fit(train_features_selected, train_labels)

# 预测（硬分类结果）
predictions = clf.predict(test_features_selected)
# 获取正类概率（用于曲线绘制，仅适用于二元分类）
try:
    probas = clf.predict_proba(test_features_selected)
    # 确保是二元分类，取类别1的概率作为正类概率
    if probas.shape[1] == 2:
        positive_probas = probas[:, 1]
        print("\n成功获取正类预测概率，将绘制曲线")
    else:
        positive_probas = None
        print("\n警告：非二元分类，无法绘制P-R/ROC曲线")
except Exception as e:
    positive_probas = None
    print(f"\n获取预测概率时出错: {e}，无法绘制曲线")

# 计算PRF值
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

print("\n模型性能:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# 计算并打印AUC（二元分类）
if positive_probas is not None:
    try:
        auc_score = roc_auc_score(test_labels, positive_probas)
        print(f"AUC:       {auc_score:.4f}")
    except Exception as e:
        print(f"计算AUC时出错: {e}")


# -------------------------- 新增：绘制P-R曲线（单模型） --------------------------
def plot_pr_curve(y_true, y_prob):
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(recall_vals, precision_vals, color='#FF6B6B', lw=2.5, label='贝叶斯模型（卡方优化）')
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('精确率-召回率曲线 (P-R Curve)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('result/23_pr_curve.png', dpi=300, bbox_inches='tight')  # 保存图片（文件名含标识）
    plt.show()


# -------------------------- 新增：绘制ROC曲线（单模型） --------------------------
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    # 绘制模型ROC曲线
    plt.plot(fpr, tpr, color='#4ECDC4', lw=2.5, label=f'贝叶斯模型（卡方优化） (AUC = {roc_auc:.4f})')
    # 绘制随机猜测基准线
    plt.plot([0, 1], [0, 1], color='#95A5A6', lw=1.5, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
    plt.title('受试者工作特征曲线 (ROC Curve)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('result/23_roc_curve.png', dpi=300, bbox_inches='tight')  # 保存图片（文件名含标识）
    plt.show()


# -------------------------- 调用函数绘制曲线（仅当获取到正类概率时） --------------------------
if positive_probas is not None:
    print("\n开始绘制P-R曲线和ROC曲线...")
    plot_pr_curve(test_labels, positive_probas)
    plot_roc_curve(test_labels, positive_probas)
else:
    print("\n因缺少正类预测概率，跳过曲线绘制")