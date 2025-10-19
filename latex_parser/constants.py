# Environments that should be kept as-is (not split into sentences)
PRESERVED_ENVIRONMENTS = {
    'theorem', 'lemma', 'corollary', 'proposition', 'definition',
    'remark', 'note', 'example', 'proof', 'longtable',
    'tikzpicture', 'figure', 'table', 'algorithm',
    'equation', 'equation*', 'align', 'align*', 
    'gather', 'gather*', 'multline', 'multline*',
    'eqnarray', 'eqnarray*', 'displaymath',
    'tabular', 'verbatim', 'lstlisting',
}      
GRAPH_ENVIRONMENTS = {'tikzpicture', 'figure', 'table', 'tabular', 'longtable'}
SPACING_ENVIRONMENTS = {'doublespace', 'singlespace', 'frontmatter'}  
DELETE_MACROS = {'label', 'footnote', 'url', 'href', 'path', 'bibliography'}
