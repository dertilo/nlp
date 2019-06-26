import json

import multiprocessing

import sys
from multiprocessing.pool import Pool

from commons import util_methods
from sqlalchemy import bindparam, Table, select, String, Column, func
from typing import Tuple, List, Any, Dict, Iterable

from sqlalchemy.engine import Connection
from sqlalchemy.orm import Query

from pytorch_util.pytorch_methods import iterate_and_time


def to_String(v):
    if isinstance(v,list):
        o =  ','.join([str(x) for x in v])
    elif isinstance(v,float):
        o =  '%0.4f'%v
    elif isinstance(v,dict):
        o = json.dumps(v)
    else:
        o = str(v)
    assert isinstance(o,str) or o is None
    return o

def bulk_update(conn:Connection,table:Table,col_name:str,ids_values:List[Tuple]):
    stmt = table.update().where(table.c.id == bindparam('obj_id')).values(**{col_name: bindparam('val')})
    conn.execute(stmt, [{'obj_id': eid, 'val': val} for eid, val in ids_values])


def add_column(engine, table_name, column:Column):
    column_name = column.compile(dialect=engine.dialect)
    column_type = column.type.compile(engine.dialect)
    engine.execute('ALTER TABLE %s ADD COLUMN %s %s' % (table_name, column_name, column_type))


def default_update_fun(val, old_row=None):
    return val

def insert_or_update(conn, table:Table,columns_to_update, rows_g:Iterable[Dict], update_fun=default_update_fun,batch_size = 10000):
    def fun(batch):
        insert_or_update_batch(conn,table,columns_to_update,batch,update_fun)
    util_methods.consume_batchwise(fun, rows_g, batch_size=batch_size)

def insert_or_update_batch(conn, table:Table,columns_to_update, rows:List[Dict], process_val_fun=default_update_fun):
    ids2values = {d.pop('id'):d for d in rows}

    rows_to_update = [d for d in conn.execute(select([table]).where(table.c.id.in_(ids2values.keys())))]
    ids_to_update = [d['id'] for d in rows_to_update]
    ids_to_insert = [eid for eid in ids2values.keys() if eid not in ids_to_update]

    if len(ids_to_insert)>0:
        conn.execute(table.insert(), [{**{'id':eid},**process_val_fun(old_row=None, val=ids2values[eid])} for eid in ids_to_insert])

    if len(rows_to_update)>0:
        stmt = table.update().\
            where(table.c.id == bindparam('obj_id')). \
            values(**{col_name: bindparam('val_' + col_name) for col_name in columns_to_update})

        def build_dict_of_processed_values(old_row):
            return {'val_' + col: processed_val for col, processed_val in
                    process_val_fun(ids2values[old_row['id']], old_row=old_row).items()}

        conn.execute(stmt, [{**{'obj_id': d['id']}, **build_dict_of_processed_values(d)}
                      for d in rows_to_update])

def process_table_batchwise(sqlalchemy_engine, q:Query, table:Table, process_fun,
                            batch_size=1000, num_processes=0,
                            initializer_fun=None, initargs=()):
    g = (batch for batch in fetchmany_sqlalchemy(sqlalchemy_engine, q, batch_size=batch_size))
    process_time = 0
    if num_processes>0:
        with sqlalchemy_engine.connect() as conn:
            with Pool(processes=num_processes, initializer=initializer_fun, initargs=initargs) as pool:
                for processed_batch, dur in iterate_and_time(pool.imap_unordered(process_fun, g)):
                    process_time+=dur
                    update_table(conn, processed_batch, table)

    else:
        with sqlalchemy_engine.connect() as conn:
            processed_g = (process_fun(batch) for batch in g)
            for processed_batch,dur in iterate_and_time(processed_g):
                process_time += dur
                update_table(conn, processed_batch, table)
    return process_time


def update_table(conn, processed_batch, table):
    columns_to_update = list(processed_batch[0].keys())
    [d.update({'obj_id': d.pop('id')}) for d in processed_batch]
    stmt = table.update(). \
        where(table.c.id == bindparam('obj_id')). \
        values(**{col_name: bindparam(col_name) for col_name in columns_to_update if col_name != 'id'})
    conn.execute(stmt, processed_batch)


def insert_or_overwrite(conn, table:Table, rows:List[Dict]):

    ids2values = {d.pop('id'):d for d in rows}
    ids_to_overwrite = [d['id'] for d in conn.execute(select([table]).where(table.c.id.in_(ids2values.keys())))]
    ids_to_insert = [eid for eid in ids2values.keys() if eid not in ids_to_overwrite]
    column_names = [c.name for c in table.columns.values() if c.name!='id']
    if len(ids_to_insert)>0:
        conn.execute(table.insert(), [{**{'id':eid},**ids2values[eid]} for eid in ids_to_insert])

    if len(ids_to_overwrite)>0:
        stmt = table.update().\
            where(table.c.id == bindparam('obj_id')).\
            values(**{col_name: bindparam('val_'+col_name) for col_name in column_names})
        conn.execute(stmt,
                     [{**{'obj_id': eid},**{'val_'+c:ids2values[eid][c] for c in column_names} } for eid in
                      ids_to_overwrite])

def insert_if_not_existing(conn, table:Table, data:Iterable, batch_size=10000):
    def insert_batch(rows):
        ids = set([d['id'] for d in rows])
        existing_ids = set([d['id'] for d in conn.execute(select([table]).where(table.c.id.in_(ids)))])
        ids_to_insert = set([eid for eid in ids if eid not in existing_ids])
        if len(ids_to_insert) > 0:
            conn.execute(table.insert(), [d for d in rows if d['id'] in ids_to_insert])

    util_methods.consume_batchwise(insert_batch, data, batch_size)

def fetcher_queue_filler(
        sqlalchemy_engine,
        queue:multiprocessing.Queue,
        query:Query,
        batch_size=10000):

    proxy = sqlalchemy_engine.execution_options(stream_results=True).execute(query)
    while True:
        batch = proxy.fetchmany(batch_size)
        if batch == []:
            queue.put(None)
            break
        queue.put(batch)
        # sys.stdout.write('\rqueue-size: %d'%queue.qsize())

def fetchmany_sqlalchemy(
        sqlalchemy_engine,
        query:Query,
        batch_size=10000):

    proxy = sqlalchemy_engine.execution_options(stream_results=True).execute(query)
    while True:
        batch = proxy.fetchmany(batch_size)
        if len(batch)>0:
            yield batch
        else:
            break

def fetch_batch_wise_queueing(q, sqlalchemy_engine, max_queue_size=3, batch_size = 10000):
    '''
    TODO: is this really good for anything??
    maybe only if data that is going through queue is small and sql-quering-time high!
    '''
    fetched_queue = multiprocessing.Queue(max_queue_size)
    process = multiprocessing.Process(
        name='fetcher',
        target=fetcher_queue_filler,
        kwargs={
            'sqlalchemy_engine':sqlalchemy_engine,
            'queue': fetched_queue,
            'query': q,
            'batch_size':batch_size
        },
    )
    process.start()
    while True:
        batch = fetched_queue.get()
        if batch is None:
            break
        for d in batch:
            yield d
    process.join()

def get_tables_by_reflection(sqlalchemy_metadata,sqlalchemy_engine):
    sqlalchemy_metadata.reflect(sqlalchemy_engine)
    return sqlalchemy_metadata.tables

def count_rows(sqlalchemy_engine,table):
    return sqlalchemy_engine.execute(select([func.count(table.c.id)])).first()[0]
