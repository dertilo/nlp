import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

sqlalchemy_base = declarative_base()
POSTGRES_URL = os.environ['POSTGRES_URL'] if 'POSTGRES_URL' in os.environ else 'postgres:whocares@localhost:5432/postgres'
sqlalchemy_engine = create_engine('postgresql://%s' %POSTGRES_URL ) if POSTGRES_URL is not None else None
sqlalchemy_base.metadata.bind = sqlalchemy_engine
