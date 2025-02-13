import datetime
from enum import Enum

from sqlalchemy import Column, Integer, String, Sequence, DateTime
from DBSupport import Base


class ServerState(Enum):

    RUNNING = '0'
    STOP = '1'


class ServerInfo(Base):

    __tablename__ = 'server_info'

    id = Column(Integer, Sequence('server_id_seq'), primary_key=True, autoincrement=True, comment='主键')
    project_name = Column(String(64), nullable=False, comment='项目名称', unique=True)
    hostname = Column(String(64), nullable=False, comment='服务器主机')
    username = Column(String(64), nullable=False, comment='登录用户名')
    password = Column(String(64), nullable=False, comment='登录密码')
    port = Column(Integer, nullable=False, default=22, comment='服务器端口')
    server_state = Column(String(1), nullable=False, default='0', comment='是否启动 0未启用 1启用')
    zip_file = Column(String(128), nullable=False, comment='安装包路径')
    install_type = Column(String(1), nullable=False, default='0', comment='安装方式 0离线安装 1在线安装 2docker安装')
    install_state = Column(String(1), nullable=False, default='0', comment='安装情况 0未安装 1已安装')
    status = Column(String(1), default='0', comment='数据状态 0启动 1禁用 2删除')
    create_time = Column(DateTime, default=datetime.datetime.now(), comment='创建时间')
    create_by = Column(String(64), comment='创建人')
    update_time = Column(DateTime, onupdate=datetime.datetime.now(), comment='更新时间')
    remarks = Column(String(500), nullable=False, default='', comment='备注信息')

