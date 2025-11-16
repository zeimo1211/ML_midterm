import jieba
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")


# 加载停用词（从 midtern.ipynb 提取）
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


# 加载数据（从 midtern.ipynb 提取，添加错误处理兼容 "贝叶斯+cart.py"）
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


# 预处理数据：分词 + 去停用词 + TextRank（从 midtern.ipynb 提取，兼容异常处理）
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
            # print(f"预处理出错: {e}，文本: '{text}'")
            seg_data.append("")
            textrank_data.append("")
    return seg_data, textrank_data


# 特征提取和选择（从 midtern.ipynb 提取，参数化以兼容不同版本）
def extract_features(train_seg, train_textrank, test_seg, test_textrank, train_labels,
                     vectorizer_params={}, k=20, print_features=True):
    combined_train = [seg + " " + textrank for seg, textrank in zip(train_seg, train_textrank)]
    combined_test = [seg + " " + textrank for seg, textrank in zip(test_seg, test_textrank)]

    vectorizer = TfidfVectorizer(**vectorizer_params)
    train_features = vectorizer.fit_transform(combined_train)
    test_features = vectorizer.transform(combined_test)

    k = min(k, train_features.shape[1])
    selector = SelectKBest(chi2, k=k)
    train_selected = selector.fit_transform(train_features, train_labels)
    test_selected = selector.transform(test_features)

    if print_features:
        feature_names = vectorizer.get_feature_names_out()
        selected_indices = selector.get_support(indices=True)
        feature_words = [feature_names[idx] for idx in selected_indices[:10]]  # 前10个兼容 "贝叶斯+cart.py"
        print("特征词：", feature_words)

    return train_selected, test_selected


# 性能评估（从 midtern.ipynb 提取）
def evaluate_model(test_labels, predictions, model_name="模型"):
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    print(f"{model_name} 性能:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    return precision, recall, f1