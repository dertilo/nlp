import stat
import subprocess
import sys
sys.path.append('.')

import sqlalchemy
import json
import os
from time import sleep
from typing import Dict

from commons import data_io
from sqlalchemy import select

from active_learning.datamanagement_methods import write_brat_annotations, DONE_ANNO, parse_anno_lines, row_to_dict, \
    annotator_human, join_all_ner_annotations, overwrite_ner_annotations
from sqlalchemy_util.sqlalchemy_base import get_sqlalchemy_base_engine
from sqlalchemy_util.sqlalchemy_methods import get_tables_by_reflection, process_table_batchwise


def collect_annotations_write_to_table(brat_path):
    anno_files = [brat_path + '/' + f for f in os.listdir(brat_path) if f.endswith('.ann')]
    eids_file_annolines = [(os.path.split(file)[1].replace('.ann', ''),file, list(data_io.read_lines(file)))
                      for file in anno_files]
    eids2annolines_to_collect = {eid:(file,anno_lines) for eid,file,anno_lines in eids_file_annolines if any([DONE_ANNO in line for line in anno_lines])}
    print('found %d ann-files to collect'%len(eids2annolines_to_collect.keys()))
    query = select([table]).where(table.c.id.in_([json.dumps(eid) for eid in eids2annolines_to_collect.keys()]))

    def process_batch_fun(batch):
        batch = [row_to_dict(d) for d in batch]
        def process_doc(doc:Dict):
            file,anno_lines = eids2annolines_to_collect[doc['id']]
            anno = parse_anno_lines(anno_lines, doc['sentences'])
            new_ner = join_all_ner_annotations(anno['ner'], doc['sentences'])
            ner_anno = overwrite_ner_annotations(doc['ner'], new_ner, annotator_human)
            return ner_anno
        return [{'id':json.dumps(d['id']),'ner':json.dumps(process_doc(d))}  for d in batch]

    process_table_batchwise(sqlalchemy_engine, query, table, process_batch_fun)
    files_to_remove = [file for file,_ in eids2annolines_to_collect.values()]
    for file in files_to_remove:
        os.remove(file)
        os.remove(file.replace('.ann', '.txt'))

    return [f for f in anno_files if f not in files_to_remove]

def id_from_filename(file):
    return json.dumps(os.path.split(file)[1].replace('.ann', ''))

if __name__ == '__main__':
    # ip = 'localhost'
    ip = '10.1.1.29'
    sqlalchemy_base,sqlalchemy_engine = get_sqlalchemy_base_engine(host=ip)
    table = get_tables_by_reflection(sqlalchemy_base.metadata,sqlalchemy_engine)['scierc']
    num_anno_docs = 10
    brat_path='/home/tilo/data/brat-data/scierc'
    import shutil
    if os.path.isdir(brat_path):
        shutil.rmtree(brat_path)
        # os.mkdir(brat_path)
    shutil.copytree('active_learning/brat_configurations',brat_path)
    # shutil.chown(brat_path, user='www-data', group='www-data')
    # os.chmod(brat_path, stat.S_IWOTH | stat.S_IROTH)
    # subprocess.call(['chmod', '777', brat_path])
    while True:
        anno_files = collect_annotations_write_to_table(brat_path)
        num_to_generate=num_anno_docs-len(anno_files)
        if num_to_generate>0:
            print('generating: %d'%num_to_generate)
            ids = [id_from_filename(f) for f in anno_files]
            write_brat_annotations(select([table]).where(sqlalchemy.not_(table.c.id.in_(ids))).limit(num_to_generate), brat_path, sqlalchemy_engine)
            subprocess.call(['chmod','-R', '777', brat_path])
            anno_files = [brat_path + '/' + f for f in os.listdir(brat_path) if f.endswith('.ann')]
            assert len(anno_files)==num_anno_docs
        else:
            sleep(10)

