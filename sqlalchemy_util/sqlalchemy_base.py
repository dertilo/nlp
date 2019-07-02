import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

sqlalchemy_base = declarative_base()
POSTGRES_URL = os.environ['POSTGRES_URL'] if 'POSTGRES_URL' in os.environ else 'postgres:postgres@localhost:5432/postgres'
sqlalchemy_engine = create_engine('postgresql://%s' %POSTGRES_URL ) if POSTGRES_URL is not None else None
sqlalchemy_base.metadata.bind = sqlalchemy_engine

def get_sqlalchemy_base_engine(POSTGRES_URL='postgres:postgres@localhost:5432/postgres',
                               ip = None):
    if ip is not None:
        POSTGRES_URL = 'postgres:postgres@%s:5432/postgres'%ip
    sqlalchemy_base = declarative_base()

    sqlalchemy_engine = create_engine('postgresql://%s' % POSTGRES_URL) if POSTGRES_URL is not None else None
    sqlalchemy_base.metadata.bind = sqlalchemy_engine
    return sqlalchemy_base,sqlalchemy_engine