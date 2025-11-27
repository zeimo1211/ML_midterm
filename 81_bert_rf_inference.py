import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
import pickle
import time
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

from common import evaluate_model

# GPU检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 函数：格式化时间显示
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def evaluate_ensemble():
    # 加载BERT模型
    print("加载BERT模型...")
    bert_model = BertForSequenceClassification.from_pretrained('./saved_bert_model')
    bert_model.to(device)

    # 加载随机森林模型
    print("加载随机森林模型...")
    with open('./saved_rf_model/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    # 加载测试数据
    with open('./test_data.pkl', 'rb') as f:
        bert_test_data = pickle.load(f)

    with open('./rf_test_data.pkl', 'rb') as f:
        rf_test_data = pickle.load(f)

    bert_test_labels = bert_test_data['test_labels']
    rf_test_labels = rf_test_data['test_labels']

    # 确保标签一致
    assert bert_test_labels == rf_test_labels, "BERT和随机森林的测试标签不一致"
    test_labels_np = np.array(bert_test_labels)

    # BERT推理
    print("进行BERT推理...")
    bert_test_inputs = bert_test_data['test_inputs']
    bert_test_masks = bert_test_data['test_masks']

    batch_size = 16
    validation_dataset = TensorDataset(bert_test_inputs, bert_test_masks)
    validation_dataloader = DataLoader(
        validation_dataset,
        sampler=SequentialSampler(validation_dataset),
        batch_size=batch_size
    )

    bert_model.eval()
    bert_predictions = []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = bert_model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        bert_predictions.append(logits)

    flat_bert_predictions = np.concatenate(bert_predictions, axis=0)
    flat_bert_predictions = np.argmax(flat_bert_predictions, axis=1).flatten()

    # 随机森林推理
    print("进行随机森林推理...")
    rf_test_features = rf_test_data['test_features']
    rf_predictions = rf_model.predict(rf_test_features)

    # 评估单独模型性能
    print("\n" + "=" * 50)
    evaluate_model(test_labels_np, flat_bert_predictions, "单独BERT模型")
    evaluate_model(test_labels_np, rf_predictions, "单独随机森林模型")

    # 集成模型
    print("\n" + "=" * 50)
    print("集成模型性能:")

    # 计算每个模型的正确预测
    bert_correct = flat_bert_predictions == test_labels_np
    rf_correct = rf_predictions == test_labels_np

    # 初始化集成模型的预测结果数组
    ensemble_predictions = np.zeros(len(test_labels_np), dtype=int)

    # 遍历每个样本，确定是否至少由一个模型正确预测
    for i in range(len(test_labels_np)):
        if bert_correct[i] or rf_correct[i]:
            ensemble_predictions[i] = test_labels_np[i]
        else:
            ensemble_predictions[i] = 1 - test_labels_np[i]  # 预测为另一个类别

    # 计算集成模型的性能指标
    ensemble_precision = precision_score(test_labels_np, ensemble_predictions)
    ensemble_recall = recall_score(test_labels_np, ensemble_predictions)
    ensemble_f1 = f1_score(test_labels_np, ensemble_predictions)

    print("Ensemble Model Precision:", ensemble_precision)
    print("Ensemble Model Recall:", ensemble_recall)
    print("Ensemble Model F1-score:", ensemble_f1)

    # 保存集成结果
    ensemble_results = {
        'bert_predictions': flat_bert_predictions,
        'rf_predictions': rf_predictions,
        'ensemble_predictions': ensemble_predictions,
        'true_labels': test_labels_np,
        'metrics': {
            'bert': evaluate_model(test_labels_np, flat_bert_predictions, "BERT"),
            'rf': evaluate_model(test_labels_np, rf_predictions, "Random Forest"),
            'ensemble': (ensemble_precision, ensemble_recall, ensemble_f1)
        }
    }

    with open('./ensemble_results.pkl', 'wb') as f:
        pickle.dump(ensemble_results, f)

    print("\n集成模型评估完成")


if __name__ == "__main__":
    evaluate_ensemble()