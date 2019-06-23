import json
import logging

import arxiv
from commons import util_methods
from sqlalchemy import Table, Column, String

from sqlalchemy_util.sqlalchemy_base import sqlalchemy_engine, sqlalchemy_base
from sqlalchemy_util.sqlalchemy_methods import insert_if_not_existing, count_rows

column_names = ['updated', 'published', 'title', 'summary', 'authors', 'links', 'arxiv_primary_category', 'tags',
                'pdf_url', 'arxiv_url', 'arxiv_comment', 'journal_reference']
columns = [Column('id', String, primary_key=True)] + [Column(colname, String) for colname in column_names]

arxiv_table = Table('arxiv', sqlalchemy_base.metadata, *columns, extend_existing=True)

logging.basicConfig(filename='arxiv_metadata_scraping.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

arxiv.logger = logging.getLogger('arxiv_metadata_scraping')

if __name__ == '__main__':

    query_string = "machine learning natural language processing"
    bulk_size = 1000
    start = count_rows(sqlalchemy_engine,arxiv_table)
    print('%d are already in postgres'%start)
    result = arxiv.query(query=query_string, id_list=[],
                         start=start,
                         max_results=1000_000,
                         sort_by="relevance",
                         sort_order="descending",
                         prune=True, iterative=True,
                         max_chunk_results=bulk_size)


    def parse_result(d):
        parsed = {k:json.dumps(d[k]) for k in ['id']+column_names}
        return parsed
    # arxiv_table.drop(sqlalchemy_engine)
    if not sqlalchemy_engine.has_table(arxiv_table.name):
        print('creating table %s' % arxiv_table.name)
        arxiv_table.create()

    with sqlalchemy_engine.connect() as conn:
        g = (parse_result(d) for d in result())
        insert_if_not_existing(conn, arxiv_table, g, batch_size=bulk_size)
