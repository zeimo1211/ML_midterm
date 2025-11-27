import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
import pickle
import time
import datetime

from common import evaluate_model

# GPU检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 函数：格式化时间显示
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def evaluate_bert():
    # 加载模型和分词器
    print("加载BERT模型...")
    model = BertForSequenceClassification.from_pretrained('./saved_bert_model')
    tokenizer = BertTokenizer.from_pretrained('./saved_bert_model')
    model.to(device)

    # 加载测试数据
    with open('./test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    test_inputs = test_data['test_inputs']
    test_masks = test_data['test_masks']
    test_labels = test_data['test_labels']

    # 创建DataLoader
    batch_size = 16
    validation_dataset = TensorDataset(test_inputs, test_masks)
    validation_dataloader = DataLoader(
        validation_dataset,
        sampler=SequentialSampler(validation_dataset),
        batch_size=batch_size
    )

    # 评估模型
    print("\n开始BERT模型验证...")
    t0 = time.time()
    model.eval()
    predictions = []

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    print(f"验证用时: {format_time(time.time() - t0)}")

    # 计算预测结果
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # 评估性能
    precision, recall, f1 = evaluate_model(test_labels, flat_predictions, "BERT模型")

    # 保存预测结果
    results = {
        'predictions': flat_predictions,
        'true_labels': test_labels,
        'metrics': {'precision': precision, 'recall': recall, 'f1': f1}
    }

    with open('./bert_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("BERT模型评估完成")


if __name__ == "__main__":
    evaluate_bert()