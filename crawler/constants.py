import json


CATEGORIES = {
    "cs": [
        "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY", 
        "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR", 
        "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS", 
        "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC", 
        "cs.SD", "cs.SE", "cs.SI", "cs.SY"
    ], 
    "eess": ["eess.AS", "eess.IV", "eess.SP", "eess.SY"],
    "econ": ["econ.EM", "econ.GN", "econ.TH"],
    "math": [
        "math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO", "math.CT", 
        "math.CV", "math.DG", "math.DS", "math.FA", "math.GM", "math.GN", "math.GR", 
        "math.GT", "math.HO", "math.IT", "math.KT", "math.LO", "math.MG", "math.MP", 
        "math.NA", "math.NT", "math.OA", "math.OC", "math.PR", "math.QA", "math.RA", 
        "math.RT", "math.SG", "math.SP", "math.ST"
    ],
    "physics": [
        "astro-ph.CO", "astro-ph.EP", "astro-ph.GA", "astro-ph.HE", "astro-ph.IM", 
        "astro-ph.SR", "cond-mat.dis-nn", "cond-mat.mes-hall", "cond-mat.mtrl-sci", 
        "cond-mat.other", "cond-mat.quant-gas", "cond-mat.soft", "cond-mat.stat-mech", 
        "cond-mat.str-el", "cond-mat.supr-con", "gr-qc", "hep-ex", "hep-lat", "hep-ph", 
        "hep-th", "math-ph", "nlin.AO", "nlin.CD", "nlin.CG", "nlin.PS", "nlin.SI", 
        "nucl-ex", "nucl-th", "physics.acc-ph", "physics.ao-ph", "physics.app-ph", 
        "physics.atm-clus", "physics.atom-ph", "physics.bio-ph", "physics.chem-ph", 
        "physics.class-ph", "physics.comp-ph", "physics.data-an", "physics.ed-ph", 
        "physics.flu-dyn", "physics.gen-ph", "physics.geo-ph", "physics.hist-ph", 
        "physics.ins-det", "physics.med-ph", "physics.optics", "physics.pop-ph", 
        "physics.soc-ph", "physics.space-ph", "quant-ph"
    ],
    "q-bio": [
        "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT", "q-bio.PE", "q-bio.QM", 
        "q-bio.SC", "q-bio.TO"
    ],
    "q-fin": [
        "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR", 
        "q-fin.RM", "q-fin.ST", "q-fin.TR"
    ],
    "stat": ["stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH"]
}

GET_TITLE_FROM_LATEX_PROMPT = """You are an AI assistant specialized in processing academic literature information. Your task is to extract specific bibliographic information from text content parsed from Latex.

Please analyze the provided LaTeX source code and extract the following information:
- Title of the paper/publication
- Source (journal name, conference name, or publication venue)
- URL/link to the paper (if available)
- ArXiv ID (if available)

Instructions:
1. Carefully examine the text content for bibliographic entries, citation commands, or any metadata sections
2. Identify and extract the requested information fields
3. If any field cannot be found in the source code, return null for that field
4. Return the results in JSON format

Output format:
{{
  "title": "extracted title or null",
  "source": "journal/conference name or null", 
  "url": "link to paper or null",
  "arxiv_id": "arXiv identifier or null"
}}

LaTeX source code to analyze:
{content}

Please provide your extraction results in the specified JSON format."""


def load_local(fn):
    with open(fn, "r+", encoding="utf-8") as f:
        d = [json.loads(line.strip()) for line in f if line.strip()]
    return d


def print_json(d, fn):
    with open(fn, "w+", encoding="utf-8") as f:
        for x in d:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def yield_local(fn):
    with open(fn, "r+", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except:
                    pass