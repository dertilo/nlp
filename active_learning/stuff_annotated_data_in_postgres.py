from commons import data_io
from sqlalchemy import Column, String, Table

from active_learning.sqlbase import sqlalchemy_base, sqlalchemy_engine
from sqlalchemy_util.sqlalchemy_methods import insert_or_overwrite, insert_or_update

if __name__ == '__main__':
    data = data_io.read_jsons_from_file('/home/tilo/code/NLP/scisci_nlp/data/scierc_data/json/train.json')
    data = ({**{'id':d.pop('doc_key')},**d} for d in data)
    data = (d for d in data if isinstance(d['id'],str))
    columns = [Column('id', String, primary_key=True)] + [Column(colname, String) for colname in ['sentences','ner','relations','clusters']]
    table_name = 'scierc'
    table = Table(table_name, sqlalchemy_base.metadata, *columns, extend_existing=True)

    # table.drop(sqlalchemy_engine)
    if not sqlalchemy_engine.has_table(table.name):
        print('creating table %s'%table.name)
        table.create()

    def update_fun(val,old_row):
        # 'TEST' + old_row[k]
        # d = {k: str(v) for k, v in val.items()}
        d = {'sentences':'TEST'+old_row['sentences']}
        return d

    with sqlalchemy_engine.connect() as conn:
        insert_or_update(conn,table,['sentences'],data,update_fun=update_fun)