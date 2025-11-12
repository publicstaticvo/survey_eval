import json
import copy
import torch
import transformers
import torch.distributed as dist
from torch.utils.data import Dataset
from typing import Any, Union, Optional
from transformers import PreTrainedTokenizer


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def load_train_eval_data(datas: str, split_test_ratio: float = 0, min_score_diff: float = 0):
    train_data, valid_data = [], []
    for fn in datas:
        data = []
        with open(fn) as f:
            for line in f:
                x = json.loads(line.strip())
                for i, topic in enumerate(x['topics']):
                    nkind = 0
                    if x['scores']['positive'][i] > x['scores']['easy_negative'][i] + min_score_diff: nkind += 1
                    if x['hard_negative']:
                        if x['scores']['positive'][i] > x['scores']['hard_negative'][i] + min_score_diff: nkind += 2
                        if x['scores']['hard_negative'][i] > x['scores']['easy_negative'][i] + min_score_diff: nkind += 4
                    else: nkind += 8
                    if nkind == 7 or nkind == 9:
                        xcopy = copy.deepcopy(x)
                        xcopy['topics'] = topic
                        xcopy['id'] = f"{x['id']}++{i}"
                        data.append(xcopy)
        # 分割测试集和训练集
        if split_test_ratio:
            valid_num = int(split_test_ratio * len(data))
            valid_data.extend(data[-valid_num:])
            train_data.extend(data[:-valid_num])
        else:
            train_data.extend(data)
    return train_data, valid_data


class P2QDataCollator:

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """
        将训练数据{"topic": M个, "positive": xxx, "hard_negative": xxx, "easy_negative": xxx}处理成M*3个输入序列。
        """
        texts = [f"{x['topics']}[SEP]{x[key]['title']}. {x[key]['abstract']}" \
                 for x in batch for key in ['positive', 'hard_negative', 'easy_negative']]
        for i in range(0, len(texts), 3): assert texts[i][:5] == texts[i + 1][:5] == texts[i + 2][:5]
        texts = self.tokenizer(texts, return_tensors='pt', padding='max_length', max_length=1024, truncation=True)
        return {"input_ids": texts.input_ids, "attention_mask": texts.attention_mask}
