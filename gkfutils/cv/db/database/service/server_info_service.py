from typing import List
from DBSupport import session
from entity.server_info import ServerInfo


class ServerInfoService:

    def insert(self, server_info: ServerInfo) -> bool:
        try:
            result = session.query(ServerInfo).filter_by(hostname=server_info.hostname).first()
            if result is not None:
                # Logger().info('数据已经存在了：' + server_info.hostname)
                print('数据已经存在了：' + server_info.hostname)
                return False
            session.add(server_info)
            session.commit()
            return True
        except Exception as e:
            # Logger().error('保存数据错误：', e)
            print('保存数据错误：', e)
            session.rollback()
            return False

    def update(self, server_info: ServerInfo) -> bool:
        try:
            result = session.query(ServerInfo).filter_by(hostname=server_info.hostname).first()
            if not result:
                # Logger().info('服务器信息不存在：' + server_info.hostname)
                print('服务器信息不存在：' + server_info.hostname)

            result.hostname = server_info.hostname
            result.project_name = server_info.project_name
            result.username = server_info.username
            result.password = server_info.password
            result.port = server_info.port
            result.install_type = server_info.install_type
            result.install_state = server_info.install_state
            result.remarks = server_info.remarks
            session.commit()
            return True
        except Exception as e:
            # Logger().error('修改数据错误：', e)
            print('修改数据错误：', e)
            session.rollback()
            return False

    def remove(self, server_id: int) -> int:
        recv = session.query(ServerInfo).filter_by(id=server_id).delete()
        session.commit()
        return recv

    def queryById(self, server_id: int) -> ServerInfo:
        return session.query(ServerInfo).filter_by(id=server_id).all()

    def queryAll(self) -> List[ServerInfo]:
        return session.query(ServerInfo).all()

