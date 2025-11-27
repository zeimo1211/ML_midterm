import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
# 新增曲线相关导入
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import warnings
import numpy as np
import time

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


print("开始数据预处理...")
train_seg, train_textrank = preprocess_data(train_data)
test_seg, test_textrank = preprocess_data(test_data)

# 结合TF-IDF特征和TextRank特征
combined_train_features = [seg + " " + textrank for seg, textrank in zip(train_seg, train_textrank)]
combined_test_features = [seg + " " + textrank for seg, textrank in zip(test_seg, test_textrank)]

# 特征选择
print("开始特征提取和选择...")
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(combined_train_features)
test_features = vectorizer.transform(combined_test_features)

# 调整k值，确保不超过特征数量
k = min(500, train_features.shape[1])  # 增加特征数量，随机森林可以处理更多特征
selector = SelectKBest(chi2, k=k)
train_features_selected = selector.fit_transform(train_features, train_labels)
test_features_selected = selector.transform(test_features)

# 打印选中的特征词
feature_names = vectorizer.get_feature_names_out()
selected_indices = selector.get_support(indices=True)
feature_words = [feature_names[idx] for idx in selected_indices]
print(f"特征选择完成，共选择 {len(feature_words)} 个特征")

# ==================== 决策树模型（基准） ====================
print("\n" + "=" * 50)
print("训练决策树模型（基准）...")
start_time = time.time()

dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(train_features_selected, train_labels)
dt_predictions = dt_clf.predict(test_features_selected)

dt_time = time.time() - start_time

# 决策树性能指标
dt_precision = precision_score(test_labels, dt_predictions)
dt_recall = recall_score(test_labels, dt_predictions)
dt_f1 = f1_score(test_labels, dt_predictions)

print("决策树模型性能:")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall:    {dt_recall:.4f}")
print(f"F1-score:  {dt_f1:.4f}")
print(f"训练时间:  {dt_time:.2f}秒")

# ==================== 随机森林模型（优化） ====================
print("\n" + "=" * 50)
print("训练随机森林模型（优化）...")
start_time = time.time()

# 使用随机森林，设置合适的参数
rf_clf = RandomForestClassifier(
    n_estimators=100,  # 树的数量
    max_depth=25,  # 最大深度，防止过拟合
    min_samples_split=5,  # 内部节点再划分所需最小样本数
    min_samples_leaf=2,  # 叶子节点最少样本数
    max_features='sqrt',  # 每次分割时考虑的特征数
    bootstrap=True,  # 使用bootstrap采样
    random_state=42,  # 随机种子
    n_jobs=-1  # 使用所有CPU核心
)

rf_clf.fit(train_features_selected, train_labels)
rf_predictions = rf_clf.predict(test_features_selected)

rf_time = time.time() - start_time

# 随机森林性能指标
rf_precision = precision_score(test_labels, rf_predictions)
rf_recall = recall_score(test_labels, rf_predictions)
rf_f1 = f1_score(test_labels, rf_predictions)

print("\n随机森林模型性能:")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1-score:  {rf_f1:.4f}")
print(f"训练时间:  {rf_time:.2f}秒")

# ==================== 性能对比 ====================
print("\n" + "=" * 50)
print("性能对比:")
print(f"{'指标':<12} {'决策树':<10} {'随机森林':<10} {'提升':<10}")
print(f"{'-' * 45}")
print(f"{'Precision':<12} {dt_precision:.4f}    {rf_precision:.4f}    {rf_precision - dt_precision:+.4f}")
print(f"{'Recall':<12} {dt_recall:.4f}    {rf_recall:.4f}    {rf_recall - dt_recall:+.4f}")
print(f"{'F1-score':<12} {dt_f1:.4f}    {rf_f1:.4f}    {rf_f1 - dt_f1:+.4f}")
print(f"{'训练时间':<12} {dt_time:.2f}s     {rf_time:.2f}s     {rf_time - dt_time:+.2f}s")

# ==================== 获取预测概率（用于曲线绘制） ====================
print("\n" + "=" * 50)
print("获取预测概率用于曲线绘制...")

# 决策树概率
try:
    dt_probas = dt_clf.predict_proba(test_features_selected)
    if dt_probas.shape[1] == 2:
        dt_positive_probas = dt_probas[:, 1]
        print("成功获取决策树正类预测概率")
    else:
        dt_positive_probas = None
        print(f"警告：决策树为{dt_probas.shape[1]}分类任务，仅支持二元分类的曲线绘制")
except Exception as e:
    dt_positive_probas = None
    print(f"获取决策树预测概率时出错: {e}")

# 随机森林概率
try:
    rf_probas = rf_clf.predict_proba(test_features_selected)
    if rf_probas.shape[1] == 2:
        rf_positive_probas = rf_probas[:, 1]
        print("成功获取随机森林正类预测概率")

        # 计算AUC
        dt_auc = roc_auc_score(test_labels, dt_positive_probas) if dt_positive_probas is not None else None
        rf_auc = roc_auc_score(test_labels, rf_positive_probas)

        print(f"\nAUC对比:")
        if dt_auc is not None:
            print(f"决策树 AUC:    {dt_auc:.4f}")
        print(f"随机森林 AUC:  {rf_auc:.4f}")
        if dt_auc is not None:
            print(f"AUC提升:       {rf_auc - dt_auc:+.4f}")

    else:
        rf_positive_probas = None
        print(f"警告：随机森林为{rf_probas.shape[1]}分类任务，仅支持二元分类的曲线绘制")
except Exception as e:
    rf_positive_probas = None
    print(f"获取随机森林预测概率时出错: {e}")


# ==================== 分离的曲线绘制函数 ====================

def plot_pr_curve_comparison(y_true, dt_proba, rf_proba):
    """单独绘制P-R曲线对比图"""
    plt.figure(figsize=(10, 8))

    # 决策树P-R曲线
    if dt_proba is not None:
        dt_precision, dt_recall, _ = precision_recall_curve(y_true, dt_proba)
        dt_auc_pr = auc(dt_recall, dt_precision)
        plt.plot(dt_recall, dt_precision, color='#E74C3C', lw=2.5,
                 label=f'决策树 (AP = {dt_auc_pr:.4f})')

    # 随机森林P-R曲线
    if rf_proba is not None:
        rf_precision, rf_recall, _ = precision_recall_curve(y_true, rf_proba)
        rf_auc_pr = auc(rf_recall, rf_precision)
        plt.plot(rf_recall, rf_precision, color='#2ECC71', lw=2.5,
                 label=f'随机森林 (AP = {rf_auc_pr:.4f})')

    plt.xlabel('召回率 (Recall)', fontsize=14)
    plt.ylabel('精确率 (Precision)', fontsize=14)
    plt.title('精确率-召回率曲线对比 (P-R Curve)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('result/33_pr_curve_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve_comparison(y_true, dt_proba, rf_proba):
    """单独绘制ROC曲线对比图"""
    plt.figure(figsize=(10, 8))

    # 决策树ROC曲线
    if dt_proba is not None:
        dt_fpr, dt_tpr, _ = roc_curve(y_true, dt_proba)
        dt_auc_roc = auc(dt_fpr, dt_tpr)
        plt.plot(dt_fpr, dt_tpr, color='#E74C3C', lw=2.5,
                 label=f'决策树 (AUC = {dt_auc_roc:.4f})')

    # 随机森林ROC曲线
    if rf_proba is not None:
        rf_fpr, rf_tpr, _ = roc_curve(y_true, rf_proba)
        rf_auc_roc = auc(rf_fpr, rf_tpr)
        plt.plot(rf_fpr, rf_tpr, color='#2ECC71', lw=2.5,
                 label=f'随机森林 (AUC = {rf_auc_roc:.4f})')

    # 随机猜测基准线
    plt.plot([0, 1], [0, 1], color='#95A5A6', lw=2, linestyle='--',
             label='随机猜测 (AUC = 0.5000)')

    plt.xlabel('假正例率 (False Positive Rate)', fontsize=14)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=14)
    plt.title('ROC曲线对比', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('result/33_roc_curve_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================== 特征重要性分析 ====================
def plot_feature_importance(rf_model, feature_names, top_n=20):
    """绘制随机森林特征重要性"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 取前top_n个重要特征
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_features))

    plt.barh(y_pos, top_importances, align='center', color='#3498DB', alpha=0.7)
    plt.yticks(y_pos, top_features)
    plt.xlabel('特征重要性', fontsize=14)
    plt.title(f'随机森林前{top_n}个重要特征', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()  # 重要性从高到低显示
    plt.grid(True, alpha=0.3, axis='x', linestyle='--')
    plt.tight_layout()
    plt.savefig('result/33_rf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    # print(f"特征重要性图已保存为 'result/33_rf_feature_importance.png'")


# ==================== 执行绘图 ====================
print("\n" + "=" * 50)
print("开始绘制分离的对比图表...")

# 绘制分离的P-R曲线
if dt_positive_probas is not None and rf_positive_probas is not None:
    print("\n绘制P-R曲线对比图...")
    plot_pr_curve_comparison(test_labels, dt_positive_probas, rf_positive_probas)

    print("\n绘制ROC曲线对比图...")
    plot_roc_curve_comparison(test_labels, dt_positive_probas, rf_positive_probas)
else:
    print("跳过曲线绘制（不满足绘制条件）")

# 绘制特征重要性
try:
    print("\n绘制特征重要性图...")
    plot_feature_importance(rf_clf, feature_words, top_n=15)
except Exception as e:
    print(f"绘制特征重要性图时出错: {e}")

print("\n" + "=" * 50)
print("随机森林优化完成！")
print("主要改进：")
print("1. 使用集成学习降低过拟合风险")
print("2. 通过多棵树投票提高泛化能力")
print("3. 提供特征重要性分析")
print("4. 综合性能对比评估")
print("5. 分离的P-R和ROC曲线对比图")