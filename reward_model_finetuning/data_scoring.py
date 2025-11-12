from sentence_transformers import CrossEncoder, util
import numpy as np
import json
import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "7"
# Load pre-trained SBERT model
model = CrossEncoder('/data/tsyu/models/MiniLM-L12-v2').cuda()

def main(filename):
    # Load data
    data = {}
    with open(filename) as f:
        for line in f:
            line = json.loads(line.strip())
            data[line['id']] = line
    batch, batch_id = [], []
    for x in data.values():
        for j, topic in enumerate(x['topics']):
            for key in ['positive', 'hard_negative', 'easy_negative']:
                if x[key]:
                    batch.append((topic, f"{x[key]['title']}. {x[key]['abstract']}"))
                    batch_id.append(f"{x['id']}-{j}-{key}")
    print(f"{len(data)} items {len(batch)} sentences to process")

    # Encode sentences to get embeddings
    scores = model.predict(batch, show_progress_bar=True, convert_to_tensor=True).tolist()

    # 填回data
    for xidjk, s in zip(batch_id, scores):
        xid, j, k = xidjk.split("-")
        x = data[xid]
        if "scores" not in x: x['scores'] = {}
        if k not in x['scores']: x['scores'][k] = [0 for _ in x['topics']]
        x['scores'][k][int(j)] = s

    with open(filename, "w+") as f:
        for x in data.values():
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


for f in ['../paper_to_query/precise.jsonl', '../paper_to_query/compare.jsonl', '../paper_to_query/train.jsonl']:
    main(f)
