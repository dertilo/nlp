import os
import sys
sys.path.append(".")

from getting_data.extract_fulltext_from_pdf.fulltext import convert



import logging

log = logging.getLogger('fulltext')


if __name__ == '__main__':
    # if len(sys.argv) <= 1:
    #     sys.exit('No file path specified')
    # path = sys.argv[1].strip()
    path = '/home/tilo/code/NLP/TOOLS/arxiv-fulltext/extractor/tests/pdfs/1702.07336.pdf'
    try:
        log.info('Path: %s\n' % path)
        log.info('Path exists: %s\n' % str(os.path.exists(path)))
        textpath = convert(path)
    except Exception as e:
        sys.exit(str(e))
