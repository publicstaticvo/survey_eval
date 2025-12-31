import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def load_local(fn):
    with open(fn, encoding="utf-8") as f:
        d = [json.loads(line.strip()) for line in f if line.strip()]
    return d


# ---------- 1. 读取 JSONL ----------
jsonl_path = 'train_letor.jsonl'          # 换成你的文件
group = []                         # 每个 query 的 doc 数
X, y, qid_map = [], [], []         # 特征、label、query_id（从 0 开始连续）
dataset = load_local(jsonl_path)
for i, js in enumerate(dataset):
    chosen = js['chosen']
    rejected = js['rejected']
    n_pos, n_neg = len(chosen), len(rejected)
    # 特征 & label
    X.extend(chosen)
    y.extend([1] * n_pos)
    X.extend(rejected)
    y.extend([0] * n_neg)
    group.append(n_pos + n_neg)        # 这个 query 的 doc 总数
    qid_map.extend([i] * (n_pos + n_neg))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
group = np.array(group, dtype=np.int32)
print('总综述数:', len(group), '总论文数:', len(X))

# ---------- 2. 训练/验证划分（按 query 切，防止泄漏） ----------
n_query = len(group)
qid_train, qid_val = train_test_split(range(n_query), test_size=0.2, random_state=42)


def slice_by_qid(qids):
    mask = np.isin(qid_map, qids)
    return X[mask], y[mask]


X_tr, y_tr = slice_by_qid(qid_train)
X_va, y_va = slice_by_qid(qid_val)


group_tr = group[qid_train]
group_va = group[qid_val]

# ---------- 3. 构造 LightGBM Dataset ----------
train_data = lgb.Dataset(X_tr, label=y_tr, group=group_tr)
valid_data = lgb.Dataset(X_va, label=y_va, group=group_va, reference=train_data)

# ---------- 4. 训练 ----------
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'verbose': 0
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=300,
    callbacks=[
        lgb.early_stopping(20),
        lgb.log_evaluation(10)
    ]
)

# ---------- 5. 保存 ----------
model.save_model('ranker.txt')
np.savez('split_info.npz', qid_train=qid_train, qid_val=qid_val, group=group)
print('模型已保存')

# ---------- 6. 预测示例：对一篇新综述打分 ----------
# def predict_one_review(model, chosen_list, rejected_list):
#     """输入特征列表，返回 (得分列表, 对应索引列表)"""
#     feats = np.array(chosen_list + rejected_list, dtype=np.float32)
#     scores = model.predict(feats)
#     idx = list(range(len(chosen_list))) + list(range(len(rejected_list)))
#     return scores, idx

# # 人造一篇新综述
# new_chosen = [[random.gauss(2, 1) for _ in range(5)] for _ in range(4)]
# new_rejected = [[random.gauss(0, 1) for _ in range(5)] for _ in range(6)]
# scores, idx = predict_one_review(model, new_chosen, new_rejected)
# ranked = sorted(zip(scores, idx), key=lambda x: x[0], reverse=True)
# print('推荐引用顺序（索引）:', [i for _, i in ranked])
