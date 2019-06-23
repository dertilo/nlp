import json

import arxiv
from commons import util_methods
from sqlalchemy import Table, Column, String

from sqlalchemy_util.sqlalchemy_base import sqlalchemy_engine, sqlalchemy_base
from sqlalchemy_util.sqlalchemy_methods import insert_if_not_existing

if __name__ == '__main__':

    query_string = "machine learning natural language processing"
    result = arxiv.query(query=query_string, id_list=[],
                         max_results=1000_000,
                         sort_by="relevance",
                         sort_order="descending",
                         prune=True, iterative=True,
                         max_chunk_results=100)

    column_names = ['updated','published','title','summary','authors','links','arxiv_primary_category','tags','pdf_url','arxiv_url','arxiv_comment','journal_reference']
    columns = [Column('id', String, primary_key=True)] + [Column(colname, String) for colname in column_names]

    table = Table('arxiv', sqlalchemy_base.metadata, *columns, extend_existing=True)

    def parse_result(d):
        parsed = {k:json.dumps(d[k]) for k in ['id']+column_names}
        return parsed
    table.drop(sqlalchemy_engine)
    if not sqlalchemy_engine.has_table(table.name):
        print('creating table %s'%table.name)
        table.create()

    with sqlalchemy_engine.connect() as conn:
        g = (parse_result(d) for d in result())
        insert_if_not_existing(conn,table,g,batch_size=100)
