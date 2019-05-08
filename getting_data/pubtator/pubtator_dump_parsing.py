import codecs
import re
from typing import List
from commons import data_io

def doc_generator(file_path):
    """
    PubTator docs are of the form:

    UID|TITLE|TEXT
    UID|ABSTRACT|TEXT
    UID   SPAN   MENTION   ENTITY_TYPE  MESH_ID
    ...

    See -- data/bioconcepts2pubtator_offsets.sample

    """
    with codecs.open(file_path, "rU", encoding="utf-8") as fp:
        lines = []
        for line in fp:
            if len(line.rstrip()) == 0:
                if len(lines) > 0:
                    # filter docs to target set
                    doc_id = re.split(r'\|', lines[0].rstrip(), maxsplit=2)
                    yield lines
                    lines = []
            else:
                lines.append(line)

def get_stable_id(doc_id):
    return "%s::document:0:0" % doc_id

def pubtator_parser(content:List[str]):
    """
        1. PMID:       PubMed abstract identifier
        2. start:      Start-char-idx of Mention
        3. end:       End-char-idx of Mention
        7. Mention:
        4. Type:       i.e., gene, disease, chemical, species, and mutation
        6. Concept ID: Corresponding database identifier

    """
    # First line is the title
    split = re.split(r'\|', content[0].rstrip(), maxsplit=2)
    doc_id = int(split[0])

    stable_id = get_stable_id(doc_id)

    doc_text = split[2]

    # Second line is the abstract
    # Assume these are newline-separated; is this true?
    # Note: some articles do not have abstracts, however they still have this line
    doc_text += ' ' + re.split(r'\|', content[1].rstrip(), maxsplit=2)[2]

    # Rest of the lines are annotations
    annos = []
    for line in content[2:]:
        anno = line.rstrip('\n').rstrip('\r').split('\t')
        if anno[3] == 'NO ABSTRACT':
            continue
        else:

            # Handle cases where no CID is provided...
            if len(anno) == 5:
                anno.append("")

            # Handle leading / trailing whitespace
            if anno[3].lstrip() != anno[3]:
                d = len(anno[3]) - len(anno[3].lstrip())
                anno[1] = int(anno[1]) + d
                anno[3] = anno[3].lstrip()

            if anno[3].rstrip() != anno[3]:
                d = len(anno[3]) - len(anno[3].rstrip())
                anno[2] = int(anno[2]) - d
                anno[3] = anno[3].rstrip()

            annos.append({
                'start_end':[anno[1],anno[2]],
                'mention':anno[3],
                'type':anno[4],
                'concept-id':anno[5],
            })

    return {'PMID':doc_id,
            'stable_id':stable_id,
            'text':doc_text,
            'annos':annos}


if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())

    file_path = home+'/data/PubTator/bioconcepts2pubtator_offsets.sample'
    g = (pubtator_parser(content) for content in doc_generator(file_path))
    data_io.write_jsons_to_file('./parsed.jsonl',g)

