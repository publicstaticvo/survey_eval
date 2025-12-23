import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# 1. 人造数据：1000 篇论文，每篇 32 维特征
#    400 篇被引用（label=1），600 篇未被引用（label=0）
# -------------------------------------------------
np.random.seed(42)
n_total = 1000
n_groups = 200
dim = 5
per_group = 5
group = np.array([per_group] * n_groups, dtype=np.int32)

X = np.random.randn(n_groups, per_group, dim)
# 简单构造：被引用论文在第一个维度上均值更大
X[:, 1, 0] += 2
y = np.zeros((n_groups, per_group), dtype=int)
y[:, 0] = 1
X = X.reshape((n_total, dim))
y = y.reshape((n_total,))

df = pd.DataFrame(X, columns=[f'f{i}' for i in range(dim)])
df['label'] = y
df['pid'] = np.arange(n_total)  # 论文 id，方便后续配对

# -------------------------------------------------
# 2. 构造 pairwise 样本
#    每个正例随机配 4 个负例，生成 (pos, neg) 对
# -------------------------------------------------
pos_df = df[df.label == 1]
neg_df = df[df.label == 0]

pairs = []
n_neg_per_pos = 4
for _, pos_row in pos_df.iterrows():
    neg_sample = neg_df.sample(n_neg_per_pos, replace=False, random_state=int(pos_row.pid))
    for _, neg_row in neg_sample.iterrows():
        # 把 (正例特征, 负例特征) 横向拼成一条样本
        pair_feats = np.concatenate([pos_row.drop(['label','pid']),
                                     neg_row.drop(['label','pid'])])
        pairs.append(pair_feats)

pair_matrix = np.vstack(pairs)
print('Pairwise 样本维度:', pair_matrix.shape)  # (1600, 64)

# -------------------------------------------------
# 3. 训练集 / 验证集划分
# -------------------------------------------------
X_pair = pair_matrix
y_pair = np.ones(len(X_pair))  # 正例在前，负例在后，label 全 1
X_tr, X_va, y_tr, y_va = train_test_split(X_pair, y_pair, test_size=0.2, random_state=42)

# -------------------------------------------------
# 4. 训练 LambdaMART（LightGBM LambdaRank）
# -------------------------------------------------
def train_lambdamart() -> lgb.Booster:
    train_data = lgb.Dataset(X_tr, label=y_tr, group=group)
    valid_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

    params = {
        'objective': 'lambdarank',   # LambdaMART
        'metric': 'ndcg',            # 可选
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
        callbacks=[lgb.early_stopping(20, verbose=True), lgb.log_evaluation(10)]
    )
    return model

model = train_lambdamart()
model.save_model("model.txt")

# -------------------------------------------------
# 5. 预测：对单篇论文打分
#    把原始单论文特征喂给模型，取前半段 32 维即可
# -------------------------------------------------
single_feat = df.drop(['label','pid'], axis=1).values
scores = model.predict(single_feat, num_iteration=model.best_iteration)

df['score'] = scores
# 按得分降序，取 Top-20 作为「应被引用」论文
top_papers = df.sort_values('score', ascending=False).head(20)
print('Top-20 应被引用论文 pid：', top_papers['pid'].tolist())
