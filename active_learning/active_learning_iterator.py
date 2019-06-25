import os

from commons import data_io
from sqlalchemy import select

from active_learning.datamanagement_methods import write_brat_annotations
from sqlalchemy_util.sqlalchemy_base import get_sqlalchemy_base_engine
from sqlalchemy_util.sqlalchemy_methods import get_tables_by_reflection

def collect_annotations(anno_files):
    for file in anno_files:
        anno_lines = list(data_io.read_lines(file))
        if any(['DoneAnnotating' in line for line in anno_lines]):
            assert False#TODO
    return anno_files

if __name__ == '__main__':
    ip = 'localhost'
    sqlalchemy_base,sqlalchemy_engine = get_sqlalchemy_base_engine(ip=ip)
    table = get_tables_by_reflection(sqlalchemy_base.metadata,sqlalchemy_engine)['scierc']
    num_anno_docs = 10
    brat_path=''
    while True:
        anno_files = [brat_path+'/'+f for f in os.listdir(brat_path) if f.endswith('.ann')]
        anno_files = collect_annotations(anno_files)
        num_to_generate=num_anno_docs-len(anno_files)
        if num_to_generate>0:
            write_brat_annotations(select([table]).limit(num_to_generate), brat_path, sqlalchemy_engine)

