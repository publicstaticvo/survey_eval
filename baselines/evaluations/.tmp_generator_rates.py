import json
from collections import defaultdict
from pathlib import Path

p = Path('survey_eval/baselines/evaluations/baselines_2_verified_weakness_taxonomy_audited.json')
d = json.load(open(p, encoding='utf-8'))
rows = d['classified_rows']
tax = {int(k): v for k, v in d['taxonomy'].items()}
source_sets = ['sgen_surveys', 'codex_surveys']
evaluators = ['llm', 'cc']

def fmt_pct(x):
    return f'{x:.1f}%'

lines = []
lines.append('# Baselines 2 Generator Problem Rates')
lines.append('')
lines.append('Scope: included classified rows in `baselines_2_verified_weakness_taxonomy_audited.json`; input-format/extraction artifacts already excluded by that file. Rates below use issue count as denominator, not paper count. The 515 pending cc_surveys rows are not included.')
lines.append('')
lines.append('## Overall by generator')
lines.append('')
lines.append('| generator | total issues | valid issues | accuracy |')
lines.append('|---|---:|---:|---:|')
for s in source_sets:
    rs = [r for r in rows if r['source_set'] == s]
    valid = sum(1 for r in rs if r['is_correct'])
    lines.append(f"| {s} | {len(rs)} | {valid} | {fmt_pct(valid/len(rs)*100 if rs else 0)} |")
lines.append('')
lines.append('## Category rates by generator')
lines.append('')
lines.append('| id | category | sgen n | sgen rate | sgen valid | sgen acc | codex n | codex rate | codex valid | codex acc |')
lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for tid in range(1,15):
    row = [str(tid), tax[tid]]
    for s in source_sets:
        rs_all = [r for r in rows if r['source_set'] == s]
        rs = [r for r in rs_all if r['taxonomy_id'] == tid]
        valid = sum(1 for r in rs if r['is_correct'])
        row += [str(len(rs)), fmt_pct(len(rs)/len(rs_all)*100 if rs_all else 0), str(valid), fmt_pct(valid/len(rs)*100 if rs else 0)]
    lines.append('| ' + ' | '.join(row) + ' |')
lines.append('')
lines.append('## Category rates by generator and evaluator')
for s in source_sets:
    for ev in evaluators:
        rs_base = [r for r in rows if r['source_set'] == s and r['evaluator'] == ev]
        lines.append('')
        lines.append(f'### {s} / {ev} (n={len(rs_base)})')
        lines.append('')
        lines.append('| id | category | n | rate | valid | acc |')
        lines.append('|---:|---|---:|---:|---:|---:|')
        for tid in range(1,15):
            rs = [r for r in rs_base if r['taxonomy_id'] == tid]
            valid = sum(1 for r in rs if r['is_correct'])
            lines.append(f"| {tid} | {tax[tid]} | {len(rs)} | {fmt_pct(len(rs)/len(rs_base)*100 if rs_base else 0)} | {valid} | {fmt_pct(valid/len(rs)*100 if rs else 0)} |")

out = Path('survey_eval/baselines/evaluations/baselines_2_generator_problem_rates.md')
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out)
