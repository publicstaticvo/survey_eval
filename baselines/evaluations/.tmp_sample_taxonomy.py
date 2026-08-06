import json
import random

p = 'survey_eval/baselines/evaluations/baselines_2_verified_weakness_taxonomy_audited.json'
rows = json.load(open(p, encoding='utf-8'))['classified_rows']
random.seed(20260801)
for tid in [11, 2]:
    pool = [r for r in rows if r['taxonomy_id'] == tid]
    sample = random.sample(pool, 20)
    print('\n### CATEGORY', tid, 'N', len(pool))
    for i, r in enumerate(sample, 1):
        print(f"[{tid}-{i}] file={r['file']} idx={r['weakness_index']} eval={r['evaluator']} src={r['source_set']} topic={r['topic']} judgment={r['judgment']} sev={r['severity']}")
        print('ISSUE:', r['issue'].replace('\n', ' '))
        print('LOCATION:', str(r.get('location', '')).replace('\n', ' ')[:500])
        print('EVIDENCE:', str(r.get('evidence', '')).replace('\n', ' ')[:600])
