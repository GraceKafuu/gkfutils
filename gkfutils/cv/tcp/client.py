import socket
import threading


class TCP_Client(object):
    def __init__(self, server_host="127.0.0.1", server_port=12345):
        self.server_host = server_host
        self.server_port = server_port

        # 创建 TCP 客户端
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 连接服务端
        self.server_address = (self.server_host, self.server_port)
        self.client_socket.connect(self.server_address)
        print(f"服务端 {self.server_address} 连接成功！")

    def run(self):
        while True:
            # 发送数据到服务端
            message = input("请输入要发送的消息: ")
            self.client_socket.send(message.encode("utf-8"))
            if not message:
                break

            # 接收服务端的响应
            response = self.client_socket.recv(1024).decode("utf-8")
            print(f"服务端回复: {response}")

        # 关闭连接
        self.close()

    def start_run(self):
        try:
            # thread.start_new_thread(self.run, ())
            t = threading.Thread(target=self.run)
            t.start()
        except Exception as e:
            print("Error: 无法启动线程 {}".format(e))

    def close(self):
        self.client_socket.close()


if __name__ == "__main__":
    tcp_client = TCP_Client(server_host="127.0.0.1", server_port=12345)
    tcp_client.start_run()
