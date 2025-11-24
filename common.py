import jieba
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")


# 加载停用词
def load_stopwords(filepath="stopwords.txt"):
    stopwords = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    except FileNotFoundError:
        print("警告：停用词文件未找到，将不使用停用词")
    return stopwords


# 加载数据
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
    return data, labels


# 预处理数据：分词 + 去停用词 + TextRank
def preprocess_data(data, stopwords, use_textrank=True):
    seg_data = []
    textrank_data = []
    for text in data:
        try:
            seg_list = [word for word in jieba.cut(text) if word not in stopwords and word.strip()]
            seg_data.append(" ".join(seg_list))

            if use_textrank:
                if text.strip():
                    keywords = jieba.analyse.textrank(text, topK=5, withWeight=False, allowPOS=('n', 'v', 'a'))
                    textrank_data.append(" ".join(keywords) if keywords else "")
                else:
                    textrank_data.append("")
            else:
                textrank_data.append("")
        except Exception as e:
            seg_data.append("")
            textrank_data.append("")
    return seg_data, textrank_data


# 改进的特征提取和选择函数
def extract_features(train_seg, train_textrank, test_seg, test_textrank, train_labels,
                     vectorizer_params={}, k=20, print_features=True):
    # 组合分词和TextRank结果
    combined_train = [seg + " " + textrank for seg, textrank in zip(train_seg, train_textrank)]
    combined_test = [seg + " " + textrank for seg, textrank in zip(test_seg, test_textrank)]

    # 设置TF-IDF参数（如果没有提供，使用默认值）
    default_params = {
        'max_features': 5000,  # 限制特征数量
        'min_df': 2,  # 忽略出现次数少于2次的词
        'max_df': 0.8,  # 忽略出现在80%以上文档中的词
        'ngram_range': (1, 2)  # 包含1-gram和2-gram
    }
    vectorizer_params = {**default_params, **vectorizer_params}

    # 使用TF-IDF向量化
    vectorizer = TfidfVectorizer(**vectorizer_params)
    train_features = vectorizer.fit_transform(combined_train)
    test_features = vectorizer.transform(combined_test)

    # 获取所有特征词
    all_feature_names = vectorizer.get_feature_names_out()

    # 特征选择
    k = min(k, train_features.shape[1])
    selector = SelectKBest(chi2, k=k)
    train_selected = selector.fit_transform(train_features, train_labels)
    test_selected = selector.transform(test_features)

    # 打印特征词
    if print_features:
        selected_indices = selector.get_support(indices=True)
        selected_features = [all_feature_names[idx] for idx in selected_indices]

        print(f"选择了 {len(selected_features)} 个最重要的特征词:")
        print("特征词列表:", selected_features)

        # 按重要性排序显示前20个特征词
        if hasattr(selector, 'scores_'):
            feature_scores = selector.scores_[selected_indices]
            sorted_indices = np.argsort(feature_scores)[::-1]  # 降序排列
            top_features = [selected_features[i] for i in sorted_indices[:min(20, len(selected_features))]]
            top_scores = [feature_scores[i] for i in sorted_indices[:min(20, len(selected_features))]]

            print("\n前20个最重要的特征词及其卡方分数:")
            for feature, score in zip(top_features, top_scores):
                print(f"  {feature}: {score:.4f}")

    return train_selected, test_selected, vectorizer, selector


# 新增函数：分析特征重要性
def analyze_feature_importance(vectorizer, selector, top_n=20):
    """分析并显示最重要的特征词"""
    if hasattr(selector, 'scores_'):
        feature_names = vectorizer.get_feature_names_out()
        selected_indices = selector.get_support(indices=True)

        if len(selected_indices) > 0:
            # 获取选中特征的分数
            feature_scores = selector.scores_[selected_indices]

            # 按分数排序
            sorted_indices = np.argsort(feature_scores)[::-1]
            top_indices = sorted_indices[:min(top_n, len(sorted_indices))]

            print(f"\n最重要的 {len(top_indices)} 个特征词:")
            for i, idx in enumerate(top_indices):
                feature_idx = selected_indices[idx]
                print(f"{i + 1}. {feature_names[feature_idx]}: {feature_scores[idx]:.4f}")

            return [feature_names[selected_indices[idx]] for idx in top_indices]

    return []


# 性能评估
def evaluate_model(test_labels, predictions, model_name="模型"):
    precision = precision_score(test_labels, predictions, average='binary', zero_division=0)
    recall = recall_score(test_labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='binary', zero_division=0)
    print(f"{model_name} 性能:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    return precision, recall, f1


# 新增函数：显示TF-IDF特征示例
def show_tfidf_features(vectorizer, data, num_samples=3):
    """显示几个样本的TF-IDF特征"""
    feature_names = vectorizer.get_feature_names_out()
    feature_matrix = vectorizer.transform(data)

    print(f"\n前{num_samples}个样本的TF-IDF特征示例:")
    for i in range(min(num_samples, len(data))):
        print(f"\n样本 {i + 1}:")
        # 获取非零特征及其值
        feature_indices = feature_matrix[i].nonzero()[1]
        feature_values = feature_matrix[i].data

        # 按值排序
        sorted_indices = np.argsort(feature_values)[::-1][:10]  # 取前10个最重要的特征
        for idx in sorted_indices:
            feature_idx = feature_indices[idx]
            print(f"  {feature_names[feature_idx]}: {feature_values[idx]:.4f}")


# 使用示例
if __name__ == "__main__":
    # 加载停用词和数据
    stopwords = load_stopwords("stopwords.txt")
    train_data, train_labels = load_data("train.txt")
    test_data, test_labels = load_data("test.txt")

    # 预处理数据
    train_seg, train_textrank = preprocess_data(train_data, stopwords, use_textrank=True)
    test_seg, test_textrank = preprocess_data(test_data, stopwords, use_textrank=True)

    # 提取特征
    vectorizer_params = {
        'max_features': 1000,
        'ngram_range': (1, 2)
    }

    train_features, test_features, vectorizer, selector = extract_features(
        train_seg, train_textrank, test_seg, test_textrank, train_labels,
        vectorizer_params=vectorizer_params, k=100, print_features=True
    )

    # 分析特征重要性
    important_features = analyze_feature_importance(vectorizer, selector, top_n=20)

    # 显示TF-IDF特征示例
    show_tfidf_features(vectorizer, train_data[:2])

    print(f"\n特征矩阵形状:")
    print(f"训练集: {train_features.shape}")
    print(f"测试集: {test_features.shape}")