import json
import re
from collections import defaultdict
from pathlib import Path

SRC = Path('survey_eval/baselines/evaluations/baselines_2_verified_weakness_taxonomy_audited.json')
OUT_JSON = Path('survey_eval/baselines/evaluations/baselines_2_taxonomy15_recoded.json')
OUT_MD = Path('survey_eval/baselines/evaluations/baselines_2_taxonomy15_final_tables.md')

data = json.load(open(SRC, encoding='utf-8'))
rows = data['classified_rows']

TAXONOMY = {
    1: 'Gap and future-work discussion insufficient',
    2: 'Comparison insufficient',
    3: 'Synthesis / original viewpoint insufficient',
    4: 'Scope / inclusion-criteria declaration missing',
    5: 'Contribution statement missing',
    6: 'Internal inconsistency',
    7: 'Hallucination',
    8: 'Missing specific references',
    9: 'Missing specific topics',
    10: 'Taxonomy or framework problem',
    11: 'Evidence support insufficient / argumentation not rigorous',
    12: 'Writing clarity and presentation',
    13: 'Visualization deficiency',
    14: 'Contribution novelty problem / venue mismatch',
    15: 'Input-format / parsing artifact',
}

RE_FLAGS = re.I | re.S

def has(text, pattern):
    return re.search(pattern, text, RE_FLAGS) is not None

def classify(row):
    issue = str(row.get('issue') or '')
    loc = str(row.get('location') or '')
    text = ' '.join([issue, loc])
    low = text.lower()
    issue_low = issue.lower()

    # 15. Input or conversion artifacts. These are not survey-content defects.
    if has(issue_low, r'\b(no readable source files|source not found|survey text unavailable|provided text|text excerpt|input conversion|latex_parser|parser|render(?:ed|ing)? artifact|input extraction|source extraction|not rendered|missing rendered|empty boxes|empty-looking|algorithm placeholders|placeholder author|author names|author information|author metadata|unknown authors\.? untitled|unknown authors|untitled\.?|reference formatting|bibtex|references\.bib|bibliography formatting|citation placeholders|<cit\.|damaged citation|table content is missing from the provided text|all references are incomplete/corrupted|all 50 references are completely unusable|no titles, authors, venues|missing from the provided text)\b'):
        # Keep genuine citation-fabrication claims out of 15 when the issue gives concrete fake/future/nonexistent references.
        if not has(issue_low, r'fabricated|non-existent|nonexistent|future-dated|ai-generated citations|citation authenticity|misattribution|does not exist|hallucinat'):
            return 15, 'input/parser artifact'

    # 7. Factual/citation hallucination and technical contradictions.
    if has(low, r'fabricated|non-existent|nonexistent|fake citation|citation authenticity|future-dated|ai-generated citation|misattribut|incorrect terminology|technically incorrect|mathematically incorrect|factually incorrect|contradict(?:s|ed) (?:the )?(?:cited|source|literature)|does not match (?:the )?(?:cited|source)|wrongly claims|false claim|hallucinat|invented citation|unverifiable citation|cannot be verified as accurate or real|citation accuracy|citation integrity|citation.*does not appear|references? do not exist'):
        return 7, 'factual/citation hallucination'

    # 4. Scope/search/inclusion protocol.
    if has(low, r'search strategy|inclusion(?:/exclusion)? criteria|exclusion criteria|inclusion criteria|scope boundar|coverage boundar|time span|venue scope|language scope|selection criteria|methodology section|systematic review protocol|corpus construction|literature selection'):
        return 4, 'scope/inclusion criteria'

    # 8/9. Missing references vs missing topics.
    if has(low, r'missing specific references|missing references|omits? (?:specific )?(?:papers?|citations?|references?)|fails to cite|does not cite|landmark papers?|seminal papers?|canonical papers?|key references?|important references?|recent references?|foundational works?|related works? missing|missing key related literature|missing related literature|without citing specific papers|lack specific references|lacks specific references'):
        # If the object is a method family/topic/benchmark rather than named papers, use missing topics.
        if has(low, r'method famil|subtopic|topics?|benchmarks?|datasets?|modalit|adversarial|robustness|fol|strips|planners?|methods? missing|key methods? missing|important recent benchmarks and methods|missing modern .*methods|missing specific recent .*advances|architectural advances|major recent advances .*missing|recent advances .*missing|does not cite or discuss specialized approaches|specialized approaches'):
            return 9, 'missing topic/method family'
        return 8, 'missing specific references'
    if has(low, r'missing specific topics|missing topics|omits? (?:a )?(?:topic|subtopic|method family|content category|benchmark|dataset|methods?)|excluded without explicit scoping|absent (?:topic|subtopic|method family)|limited discussion of .*adversarial conditions|adversarial conditions|important recent benchmarks and methods are excluded|key methods? missing'):
        # User explicitly wants adversarial-condition performance discussion as comparison if phrased as performance under conditions.
        if has(low, r'perform under|performance under|compare.*adversarial|compar.*adversarial'):
            return 2, 'comparison under conditions'
        return 9, 'missing topic/method family'

    # 14. Novelty/redundancy/venue fit, including comparison with prior surveys for novelty.
    if has(low, r'novelty|redundan|overlap with prior surveys|prior surveys|distinct organizing perspective|venue fit|venue mismatch|not sufficiently novel|contribution novelty|strong novelty claims|first .*survey|first .*taxonomy'):
        return 14, 'novelty/venue fit'

    # 6. Internal contradiction or promise-body mismatch.
    if has(low, r'internal inconsist|contradict|promise.*not|title .*absent|title mentions|claimed .* but|claims .* but|abstract claims .* but|section title.*not|not fulfilled|mismatch between|inconsistent with|count mismatch|fewer than .* references'):
        return 6, 'internal inconsistency'

    # 10. Taxonomy/framework design problems.
    if has(low, r'taxonomy|framework|categorization|classification scheme|granularity|organizing framework|conceptual framework|categories (?:are|seem)|overly broad|not mutually exclusive|orthogonal'):
        # If wording is lack of taxonomy/synthesis rather than flawed taxonomy, defer to synthesis.
        if has(low, r'lacks? .*taxonomy|no .*taxonomy|without .*taxonomy'):
            return 3, 'missing synthesis/taxonomy'
        return 10, 'taxonomy/framework problem'

    # 2. Comparison/evaluation across methods, systems, datasets, benchmarks, dimensions.
    if has(low, r'compar|contrast|benchmark performance|performance analysis|quantitative|empirical results|summary table|comparison table|benchmark summary|cross-method|across methods|across papers|across benchmarks|baselines?|metrics? across|relative effectiveness|which methods|method performance|performance table|accuracy|runtime|latency|throughput|memory|flops|shared benchmarks|adversarial conditions'):
        # Pure table readability should be visualization; but tables for performance/benchmark comparison stay here.
        if has(low, r'overlapping labels|cramped|axis|caption.*unclear|figure') and not has(low, r'performance|benchmark|method|metrics|accuracy|runtime|latency|throughput'):
            return 13, 'visualization problem'
        return 2, 'comparison insufficient'

    # 13. Figures/tables/visual summaries.
    if has(low, r'figure|figures|table|tables|diagram|visual|visualization|chart|caption|readability of .*figure|summary figure|summary table|no tables|no figures|lacks? visual'):
        return 13, 'visualization deficiency'

    # 1. Gaps/future work/open challenges.
    if has(low, r'future work|future directions|open problems|open challenges|research gaps|limitations of current work|unresolved challenges|gaps section|limitations section'):
        return 1, 'gap/future-work discussion'

    # 5. Survey contribution statement.
    if has(low, r'contribution statement|does not explicitly state (?:its )?contribution|no explicit contribution|survey contribution|what it contributes|contributions are unclear'):
        return 5, 'contribution statement missing'

    # 3. Synthesis/original perspective.
    if has(low, r'synthesis|synthesize|integrat(?:e|ion)|original viewpoint|authorial perspective|trend analysis|mainly lists|list-like|descriptive list|catalogue|catalog|lacks? .*perspective|coherent forward-looking vision|fails to synthesize'):
        return 3, 'synthesis/original viewpoint'

    # 12. Writing/presentation/organization/audience/abstract/repetition/flow.
    if has(low, r'writing|clarity|presentation|readability|grammar|language|organization|logical flow|audience|abstract|introduction|conclusion|transition|abrupt|repetition|repetitive|redundant|vague|unclear|poorly organized|hard to follow|section clarity|too brief|fragmented reading experience|many short paragraphs|narrow subsections|under-developed conclusion'):
        return 12, 'writing clarity/presentation'

    # 11. Argument support/depth/nuance/rigor.
    if has(low, r'evidence|support|unsupported|overclaim|overstates|indisputable|nuance|limitations or conditions|conditions under which|depth|deep(?:er)? technical treatment|shallow|superficial|critical rigor|analysis depth|argument|justify|substantiat|lacks? rigor|insufficiently grounded|underdeveloped treatment|unbalanced treatment'):
        return 11, 'evidence/argument support'

    # Conservative fallback: keep previous audited class if no strong signal.
    tid = int(row.get('taxonomy_id') or 11)
    if tid < 1 or tid > 14:
        tid = 11
    return tid, 'fallback to previous audited category'

recoded = []
changes = defaultdict(int)
for row in rows:
    new_id, reason = classify(row)
    item = dict(row)
    item['old_taxonomy_id'] = int(row['taxonomy_id'])
    item['old_taxonomy_category'] = row['taxonomy_category']
    item['taxonomy_id'] = new_id
    item['taxonomy_category'] = TAXONOMY[new_id]
    item['recoding_reason'] = reason
    item['is_effective_survey_issue'] = bool(row.get('is_correct')) and new_id != 15
    if item['old_taxonomy_id'] != new_id:
        changes[(item['old_taxonomy_id'], new_id)] += 1
    recoded.append(item)

# Metrics.
def group_metrics(base_rows, key_fields, include_total_valid=True):
    groups = defaultdict(list)
    for r in base_rows:
        key = tuple(r[k] for k in key_fields)
        groups[key].append(r)
    out = []
    for key in sorted(groups):
        rs = groups[key]
        total = len(rs)
        valid = sum(1 for r in rs if r['is_effective_survey_issue'])
        out.append({
            **{field: value for field, value in zip(key_fields, key)},
            'total_issues': total,
            'valid_issues': valid,
            'accuracy': round(valid / total * 100, 2) if total else 0.0,
        })
    return out

source_sets = ['sgen_surveys', 'codex_surveys']
generator_rows = [r for r in recoded if r['source_set'] in source_sets]

def fmt_pct(x):
    return f'{x:.1f}%'

lines = []
lines.append('# Baselines 2 Final Taxonomy-15 Tables')
lines.append('')
lines.append('Taxonomy 15 is `Input-format / parsing artifact`; it is not counted as an effective survey weakness. Valid/effective counts use existing manual/meta validity labels, except taxonomy 15 is forced to invalid for survey-quality metrics.')
lines.append('')
lines.append('Rows included: {} classified rows. Pending cc_surveys rows from prior run remain excluded.'.format(len(recoded)))
lines.append('')
lines.append('## Table 1. sgen / codex problem frequency')
lines.append('')
lines.append('| id | category | sgen count | sgen rate | codex count | codex rate |')
lines.append('|---:|---|---:|---:|---:|---:|')
for tid in range(1,16):
    cells = [str(tid), TAXONOMY[tid]]
    for s in source_sets:
        base = [r for r in generator_rows if r['source_set'] == s]
        n = sum(1 for r in base if r['taxonomy_id'] == tid)
        cells.extend([str(n), fmt_pct(n / len(base) * 100 if base else 0)])
    lines.append('| ' + ' | '.join(cells) + ' |')

lines.append('')
lines.append('## Table 2. detector + prompt effective frequency and accuracy')
lines.append('')
lines.append('Each cell is `valid / total (accuracy)`. This table uses all recoded classified rows, not only sgen/codex.')
combos = sorted(set((r['evaluator'], r['prompt']) for r in recoded))
header = ['id', 'category'] + [f'{ev}+{pr}' for ev, pr in combos]
lines.append('| ' + ' | '.join(header) + ' |')
lines.append('|---:|---' + '|---:' * len(combos) + '|')
for tid in range(1,16):
    cells = [str(tid), TAXONOMY[tid]]
    for ev, pr in combos:
        rs = [r for r in recoded if r['evaluator'] == ev and r['prompt'] == pr and r['taxonomy_id'] == tid]
        total = len(rs)
        valid = sum(1 for r in rs if r['is_effective_survey_issue'])
        cells.append(f'{valid}/{total} ({fmt_pct(valid/total*100 if total else 0)})')
    lines.append('| ' + ' | '.join(cells) + ' |')

lines.append('')
lines.append('## Recoding change summary')
lines.append('')
lines.append('| old -> new | count |')
lines.append('|---|---:|')
for (old, new), n in sorted(changes.items(), key=lambda x: (-x[1], x[0])):
    lines.append(f'| {old} {TAXONOMY.get(old, "?")} -> {new} {TAXONOMY[new]} | {n} |')

out = {
    'status': 'taxonomy15_recoded_from_audited_rows',
    'source': str(SRC),
    'taxonomy': TAXONOMY,
    'summary': {
        'input_rows': len(rows),
        'recoded_rows': len(recoded),
        'taxonomy15_rows': sum(1 for r in recoded if r['taxonomy_id'] == 15),
        'changed_rows': sum(1 for r in recoded if r['old_taxonomy_id'] != r['taxonomy_id']),
    },
    'rows': recoded,
}
OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(out['summary'], ensure_ascii=False, indent=2))
print(OUT_MD)
print(OUT_JSON)






