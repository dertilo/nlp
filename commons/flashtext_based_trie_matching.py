import collections
from time import time
from typing import Iterable

import numpy
from flashtext import KeywordProcessor

from commons import data_io


def get_nrams(w,min_n=3,max_n=5):
    return [w[k:k+ngs] for ngs in range(min_n,max_n+1) for k in range(len(w)-ngs) ]

def build_flashtext_trie(file,limit=numpy.Inf):
    id2phrases = {}

    def process_line(line):
        line = line.replace('\n', '')
        if '\t' in line:
            id, phrase = line.split('\t')
            if id in id2phrases:
                id2phrases[id].append(phrase)
            else:
                id2phrases[id] = [phrase]
        else:
            phrase = line
            id2phrases[len(id2phrases)+1]=[phrase]

    [process_line(line) for line in data_io.read_lines(file,limit=limit)]
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_dict(id2phrases)
    # print(keyword_processor.extract_keywords('gesellschaftervertrag'))
    return keyword_processor

class FlashTextMatcher(object):
    def __init__(self, file=None, phrases_leaves_g:Iterable=None, limit=numpy.Inf,case_sensitive = True) -> None:
        start = time()
        if file is not None:
            self._kwp = build_flashtext_trie(file,limit)
        elif phrases_leaves_g is not None:
            self._kwp = KeywordProcessor(case_sensitive=case_sensitive)
            [self._kwp.add_keyword(phrase,leaf) for phrase,leaves in phrases_leaves_g for leaf in leaves]
        else:
            assert False
        print('took %0.2f seconds to stuff %d phrases in trie' % (time()-start,len(self._kwp)))

    def get_matches(self,s:str):
        s.lower()
        return self._kwp.extract_keywords(s)


# def build_dump_dictionary( f = '/tmp/ngrams.csv',min_freq=9,limit = 100000):
#     from sqlalchemy import Table, Column, String, select
#     from commons_old.sql_database_stuff.base import sqlalchemy_base, session_supplier
#     metadata = sqlalchemy_base.metadata
#     company = Table('companies_no_content', metadata,
#                     Column('id', String, primary_key=True),
#                     Column('name', type_=String),
#                     Column('purpose', type_=String),
#                     # Column('buergel_industries', type_=String),
#                     extend_existing=True
#                     )
#     session, engine = session_supplier()
#     conn = engine.connect()
#
#     def com_dict_g():
#         c = 0
#         q_com = select([company]).order_by(company.c.id.asc()).limit(limit)
#         for d in conn.execute(q_com):
#             c += 1
#             yield d['purpose']
#
#     c = collections.Counter((w.lower() for s in com_dict_g() for _, _, w in regex_tokenizer(s)))
#     print('dict-size: %d' % len(c))
#     print(len([k for k, f in c.items() if f >= min_freq]))
#     data_io.write_to_file(f, (k for k, f in c.items() if f >=min_freq))

