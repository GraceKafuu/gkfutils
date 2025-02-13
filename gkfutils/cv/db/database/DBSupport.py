from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import os 

Base = declarative_base()

# 相对路径还不成功？暂时还不清楚原因。2025.02.13
CONFIG_FOLDER = 'D:/GraceKafuu/Python/github/gkfutils/gkfutils/cv/db/database/database'

engine = create_engine(
    f'sqlite:///{CONFIG_FOLDER}/env_manage.db', max_overflow=0, pool_size=5, pool_timeout=30, pool_recycle=60, echo=True)

# 绑定引擎
Session = sessionmaker(bind=engine)
# 创建数据连接池，直接使用session即可为当前线程拿出一个连接对象conn
# 内部会采用threading.local进行隔离
session = scoped_session(Session)


def init_table():
    Base.metadata.create_all(engine)
