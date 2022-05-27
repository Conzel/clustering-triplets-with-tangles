# Latex readme
Compiling can be done as always in Latex. We use a special style file for our bibentries, `thesis.bst`. 
This style file is based on the natbib style and abbreviates author names, omits DOIs/URLs and inserts the appropriate formatting
for arXiv files.

To make use of the arXiv functionality, make sure to mark the items as "Preprint" in Zotero. This exports them as "@misc" type, 
with appropriate arXiv links (id, archivename, ...). This is rendered by the .bst file as follows: _arXiv preprin arXiv:<arXiv id>_

Due to the implementation, currently we can only have arXiv preprint. If another online repository becomes widely used, I will have to learn 
the weird, unnamed postfix-language that .bst files are implemented with. 
