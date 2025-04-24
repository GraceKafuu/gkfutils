import socket
import threading
import time



class TCP_Server(object):
    def __init__(self, host="127.0.0.1", port=12345):
        self.host = host
        self.port = port
        self.server_is_closed = False
        self.client_is_closed = False

    def run(self):
        while True:
            try:
                # 创建 TCP 服务端
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

                # 绑定 IP 地址和端口
                self.server_address = (self.host, self.port)  # 绑定本机 IP 和端口 12345
                self.server_socket.bind(self.server_address)

                # 监听客户端连接
                self.server_socket.listen(5)  # 最大连接数为 5
                print(f"服务端 {self.server_address} 已启动, 等待客户端连接...")
                
                # 接受客户端连接
                self.client_socket, self.client_address = self.server_socket.accept()
                print(f"客户端 {self.client_address} 连接成功！")

                while True:
                    # 接收客户端发送的数据
                    data = self.client_socket.recv(1024).decode("utf-8")
                    self.recv_data = data
                    if not data:
                        self.client_close()
                        self.server_close()
                        break

                    print(f"收到客户端发送的消息: {data}")

                    # 发送数据给客户端
                    response = f"服务端已收到你的消息: {data}"
                    self.client_socket.send(response.encode("utf-8"))

                # 关闭连接
                self.client_close()
                self.server_close()
            except Exception as e:
                print(e)

            finally:
                self.client_close()
                self.server_close()

    def start_run(self):
        try:
            t = threading.Thread(target=self.run, args=())
            t.start()
        except Exception as e:
            print("Error: 无法启动线程 {}".format(e))

    def server_close(self):
        if not self.server_is_closed:
            self.server_socket.close()
            self.server_is_closed = True
    
    def client_close(self):
        if not self.client_is_closed:
            self.client_socket.close()
            self.client_is_closed = True





if __name__ == "__main__":
    tcp_server = TCP_Server(host="127.0.0.1", port=12345)
    tcp_server.start_run()




