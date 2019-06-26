import json

from commons import data_io

from sqlalchemy_util.sqlalchemy_base import get_sqlalchemy_base_engine
from sqlalchemy_util.sqlalchemy_methods import insert_or_update, get_tables_by_reflection

if __name__ == '__main__':
    # ip = 'localhost'
    ip = '10.1.1.29'
    sqlalchemy_base,sqlalchemy_engine = get_sqlalchemy_base_engine(ip=ip)

    data = data_io.read_jsons_from_file('/home/tilo/code/NLP/scisci_nlp/data/scierc_data/json/train.json')
    data = ({**{'id':json.dumps(d.pop('doc_key'))},**d} for d in data)
    data = (d for d in data if isinstance(d['id'],str))
    table = get_tables_by_reflection(sqlalchemy_base.metadata,sqlalchemy_engine)['scierc']

    # columns = [Column('id', String, primary_key=True)] + [Column(colname, String) for colname in ['sentences','ner','relations','clusters']]
    # table = Table('scierc', sqlalchemy_base.metadata, *columns, extend_existing=True)

    # table.drop(sqlalchemy_engine)
    if not sqlalchemy_engine.has_table(table.name):
        print('creating table %s'%table.name)
        table.create()

    def update_fun(val,old_row):
        d = {k: json.dumps({'annotator_luan':v}) for k, v in val.items() if k!='sentences'}
        d['sentences']=json.dumps(val['sentences'])
        return d

    with sqlalchemy_engine.connect() as conn:
        insert_or_update(conn,table,['sentences','ner','relations','clusters'],data,update_fun=update_fun)