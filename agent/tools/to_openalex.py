import re

def strip_outer_parentheses(s: str) -> str:
    """
    递归去掉包住整个表达式的最外层括号
    """
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        valid = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and i != len(s) - 1:
                valid = False
                break
        if valid:
            s = s[1:-1].strip()
        else:
            break
    return s


def clean_term(term: str) -> str:
    term = term.replace('"', '').replace("'", '')
    term = re.sub(r"[^\w\s\-_]", " ", term)
    term = re.sub(r"\s+", " ", term).strip()
    return term


def split_top_level_and(query: str):
    parts = []
    depth = 0
    buffer = []

    tokens = re.split(r"(\bAND\b)", query, flags=re.IGNORECASE)

    for tok in tokens:
        if "(" in tok:
            depth += tok.count("(")
        if ")" in tok:
            depth -= tok.count(")")

        if tok.upper() == "AND" and depth == 0:
            parts.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(tok)

    if buffer:
        parts.append("".join(buffer).strip())

    return parts


def split_or(group: str):
    group = strip_outer_parentheses(group)
    terms = re.split(r"\bOR\b", group, flags=re.IGNORECASE)
    return [clean_term(t) for t in terms if clean_term(t)]


def to_openalex(query: str) -> str:
    query = strip_outer_parentheses(query)
    and_groups = split_top_level_and(query)

    blocks = []
    for group in and_groups:
        or_terms = split_or(group)
        if or_terms:
            blocks.append("|".join(or_terms))

    return ",default.search:".join(blocks)
    # return text.replace('default.search:', '', 1)


if __name__ == "__main__":
    string = ['relationship sexuality education programmes intellectual disabilities design delivery', '(("sexuality education" OR "sexual health education") AND ("intellectual disability" OR "intellectual disabilities") AND (programme OR program))', '(("curriculum development" OR "instructional design" OR "program development") AND ("sexuality education" OR "sexual health education") AND ("intellectual disability" OR "intellectual disabilities") AND ("delivery method" OR "teaching strategy" OR intervention))', '(("theoretical framework" OR model OR principle) AND ("sexuality education" OR "sexual health education") AND ("intellectual disability" OR "intellectual disabilities") AND (rights OR inclusion OR empowerment OR "person centred"))', '(("sexuality education" OR "sexual health education") AND ("intellectual disability" OR "intellectual disabilities") AND (school OR college OR "community setting" OR "residential care" OR "health service"))', '(("systematic review" OR "meta-analysis" OR review) AND ("sexuality education" OR "sexual health education") AND ("intellectual disability" OR "intellectual disabilities") AND (guideline OR standard OR policy OR "best practice"))']
    
    for s in string:
        print(s)
        print(to_openalex(s))
