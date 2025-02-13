from entity.server_info import ServerInfo
from service.server_info_service import ServerInfoService
from datetime import datetime


if __name__ == '__main__':
    si = ServerInfo()
    si.id = 5
    si.project_name = "Test2服务器"
    si.hostname = '10.10.11.209'
    si.username = 'gx'
    si.password = '123'
    si.port = 22
    si.server_state = '1'
    si.status = '0'
    si.create_time = datetime.now()
    si.create_by = ''
    si.update_time = datetime.now()
    si.zip_file = 'ubuntu-install/guoxun.tar.gz'
    si.install_type = '1'
    si.install_state = '0'
    si.remarks = ''

    sis = ServerInfoService()
    sis.insert(si)



