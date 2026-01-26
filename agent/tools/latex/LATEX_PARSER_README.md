# LaTeX Paper Parser Documentation

## Overview

The `latex_parser.py` module provides a comprehensive parser for converting LaTeX academic papers (`.tex` files) into a structured `Paper` object. This parser handles:

- **Section hierarchy** (sections, subsections, subsubsections)
- **Content extraction** with proper sentence segmentation
- **Citation tracking** from citations in the text
- **Bibliography parsing** from `.bib`, `.bbl`, and `\thebibliography` environments
- **Special environment handling** (equations, figures, tables, algorithms)
- **`\input` command processing** for multi-file papers

## Installation

```bash
pip install pylatexenc
```

## Core Classes

### `Paper` (dataclass)
Represents an entire academic paper with hierarchical structure.

**Attributes:**
- `title: str` - Paper title
- `author: Optional[str]` - Author information
- `abstract: Optional[Section]` - Abstract section
- `children: List[Section]` - Top-level sections
- `paragraphs: List[Paragraph]` - Paragraphs in preamble/main body
- `references: dict` - Bibliography entries mapped by citation key
- `has_section_index: bool` - Whether to include section numbering

**Methods:**
- `get_skeleton() -> dict` - Convert to JSON-serializable format

### `Section` (dataclass)
Represents a section or subsection with hierarchical nesting.

**Attributes:**
- `name: str` - Section title
- `level: int` - Nesting level (0=section, 1=subsection, 2=subsubsection)
- `paragraphs: List[Paragraph]` - Paragraphs in this section
- `children: List[Section]` - Nested subsections
- `father: Optional[Union[Section, Paper]]` - Parent section

**Methods:**
- `add_paragraph(Paragraph)` - Add a paragraph to this section
- `add_child(Section)` - Add a subsection
- `get_skeleton(section_id) -> dict` - Convert to JSON-serializable format

### `Paragraph` (dataclass)
Represents a paragraph containing multiple sentences.

**Attributes:**
- `sentences: List[Sentence]` - Sentences in this paragraph
- `environment_type: EnvironmentType` - Type of content (text, equation, figure, etc.)
- `father: Optional[Section]` - Parent section

**Methods:**
- `add_sentence(Sentence)` - Add a sentence to this paragraph
- `get_skeleton() -> List[dict]` - Convert to JSON-serializable format

### `Sentence` (dataclass)
Represents a single sentence with associated citations.

**Attributes:**
- `text: str` - Sentence text (without LaTeX commands)
- `citations: List[str]` - Citation keys referenced in this sentence
- `environment_type: EnvironmentType` - Type of environment
- `father: Optional[Paragraph]` - Parent paragraph

**Methods:**
- `get_skeleton() -> dict` - Convert to JSON-serializable format

### `EnvironmentType` (enum)
Enumerates the types of content environments:

- `TEXT` - Regular text paragraph
- `EQUATION` - Mathematical equations
- `FIGURE` - Figures and images
- `TABLE` - Tables and tabular environments
- `ALGORITHM` - Algorithm descriptions
- `LISTING` - Code listings
- `VERBATIM` - Verbatim text
- `THEOREM` - Theorem/proof environments
- `UNKNOWN` - Unknown environment type

## Main Class: `LaTeXParser`

### Initialization
```python
from latex_parser import LaTeXParser

parser = LaTeXParser("path/to/main.tex")
paper = parser.parse()
```

### Key Methods

#### `parse() -> Paper`
Main parsing method that:
1. Reads the main LaTeX file
2. Extracts metadata (title, author)
3. Processes `\input` commands
4. Parses document structure (sections, paragraphs)
5. Extracts references from bibliography

#### `_process_input_commands(content: str) -> str`
Processes `\input{filename}` and `\include{filename}` commands by:
- Reading referenced `.tex` files
- Recursively processing nested inputs
- Preserving file tracking to avoid loops

#### `_parse_sections(content: str, parent: Section)`
Recursively parses section hierarchy using regex pattern matching:
```
\section{...}
\subsection{...}
\subsubsection{...}
```

#### `_parse_text_into_section(content: str, section: Section)`
Processes paragraph text by:
- Removing LaTeX comments (lines starting with `%`)
- Splitting into sentences
- Extracting citations
- Detecting environment types

#### `_extract_citations(text: str) -> List[str]`
Extracts citation keys from LaTeX commands:
- `\cite{key}`
- `\citep{key}`
- `\citet{key}`
- Supports multiple keys: `\cite{key1,key2}`

#### `_extract_references(content: str, paper: Paper)`
Handles bibliography in multiple formats:
1. BibTeX files (`.bib`)
2. Compiled bibliography (`.bbl`)
3. LaTeX `\thebibliography` environment

#### `_clean_latex(text: str) -> str`
Removes LaTeX commands and formatting:
- Removes `\textbf{...}`, `\textit{...}`, etc.
- Removes citations
- Removes inline math (`$...$`)
- Removes special characters and commands

## Usage Examples

### Basic Parsing
```python
from latex_parser import parse_paper

# Parse a paper
paper = parse_paper("path/to/main.tex")

# Access basic info
print(f"Title: {paper.title}")
print(f"Sections: {len(paper.children)}")
print(f"References: {len(paper.references)}")
```

### Traverse Sections
```python
def print_sections(section, indent=0):
    print("  " * indent + f"- {section.name}")
    for child in section.children:
        print_sections(child, indent + 1)

for section in paper.children:
    print_sections(section)
```

### Extract Content
```python
# Get all sentences with their citations
for section in paper.children:
    for para in section.paragraphs:
        for sentence in para.sentences:
            print(f"Text: {sentence.text}")
            print(f"Citations: {sentence.citations}")
            print(f"Type: {sentence.environment_type.value}")
```

### Analyze Citations
```python
# Count citation usage
citation_counts = {}
for section in paper.children:
    for para in section.paragraphs:
        for sentence in para.sentences:
            for cite in sentence.citations:
                citation_counts[cite] = citation_counts.get(cite, 0) + 1

# Find most cited papers
top_cited = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)
for cite, count in top_cited[:10]:
    print(f"{cite}: {count} citations")
```

### Serialize to JSON
```python
import json

# Get skeleton structure
skeleton = paper.get_skeleton()

# Save to file
with open("paper.json", "w") as f:
    json.dump(skeleton, f, indent=2, default=str)
```

### Search Content
```python
# Find sentences mentioning "deep learning"
for section in paper.children:
    for para in section.paragraphs:
        for sentence in para.sentences:
            if "deep learning" in sentence.text.lower():
                print(f"[{section.name}] {sentence.text}")
```

### Filter by Environment Type
```python
from latex_parser import EnvironmentType

# Extract all equations
for section in paper.children:
    for para in section.paragraphs:
        if para.environment_type == EnvironmentType.EQUATION:
            print(f"Equation in {section.name}: {para.sentences[0].text}")
```

## Limitations and Notes

1. **LaTeX Complexity**: The parser uses regex and pylatexenc, so it may not handle all complex LaTeX constructs perfectly. Advanced macros and custom commands might not be fully parsed.

2. **Math Expressions**: Inline math (`$...$`) is removed from sentence text. Display math is partially preserved but not fully parsed.

3. **Comments**: LaTeX comments (`%`) are removed. Multi-line comments might not be fully handled.

4. **Encoding**: Files are read with UTF-8 encoding, with fallback to latin-1 for compatibility.

5. **Bibliography Format**: The parser works best with standard BibTeX or `.bbl` files. Complex bibliography styles might not parse perfectly.

6. **File References**: The `\input` command processor only looks in the same directory as the main file.

## Performance Considerations

- **Large Documents**: For very large papers with many sections, parsing might take a few seconds.
- **Memory**: The parser loads the entire file into memory.
- **Regex Performance**: Complex LaTeX patterns might slow down parsing.

## Testing

Run the included test script:
```bash
python test_parser.py
```

Run example usage demonstrations:
```bash
python example_usage.py
```

## Architecture Highlights

1. **Two-Pass Approach**: 
   - First pass: Extract metadata and structure
   - Second pass: Parse content into paragraphs and sentences

2. **Environment Detection**: Automatically detects special environments (equations, figures, tables) to tag content appropriately.

3. **Citation Tracking**: Maintains a 1:1 mapping between citations in text and sentences, enabling citation-based content analysis.

4. **Hierarchical Structure**: Preserves parent-child relationships throughout the document tree, enabling navigation and tree traversal.

5. **Extensibility**: Can be extended to handle custom LaTeX commands and environments by modifying the pattern lists.

## Future Enhancements

- [ ] Better equation parsing and representation
- [ ] Support for custom LaTeX commands
- [ ] Smarter paragraph boundary detection
- [ ] Cross-reference resolution
- [ ] Figure caption extraction
- [ ] Multi-language support
- [ ] Performance optimization for very large documents
