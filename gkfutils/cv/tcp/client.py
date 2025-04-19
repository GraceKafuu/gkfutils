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

        self.message = None

    def run(self):
        while True:
            # # 发送数据到服务端
            # message = input("请输入要发送的消息: ")

            if self.message is None: continue

            self.client_socket.send(self.message.encode("utf-8"))
            if not self.message:
                break

            # 接收服务端的响应
            response = self.client_socket.recv(1024).decode("utf-8")
            print(f"服务端回复: {response}")

        # 关闭连接
        self.close()

    def start_run(self):
        try:
            t = threading.Thread(target=self.run)
            t.start()
        except Exception as e:
            print("Error: 无法启动线程 {}".format(e))

    def close(self):
        self.client_socket.close()

    # def send_msg(self, msg):
    #     self.client_socket.send(msg.encode("utf-8"))

    # def start_send_msg(self, msg):
    #     try:
    #         t = threading.Thread(target=self.send_msg, args=(msg,))
    #         t.start()
    #     except Exception as e:
    #         print("Error: 无法启动线程 {}".format(e))



if __name__ == "__main__":
    # tcp_client = TCP_Client(server_host="127.0.0.1", server_port=12345)
    # tcp_client.start_run()
    # tcp_client.start_send_msg("OK")

    # tcp_client.client_socket.send("OK".encode("utf-8"))
    # tcp_client.close()

    # tcp_client.message = "OK!!"
    # import time
    # time.sleep(1)
    # tcp_client.message = None
    # time.sleep(1)
    # tcp_client.message = "==========="
    # # tcp_client.close()


    tcp_client = TCP_Client(server_host="127.0.0.1", server_port=12345)

    import time

    i = 0
    while True:
        i += 1

        i_str = str(i)
        tcp_client.client_socket.send(i_str.encode("utf-8"))
        # 接收服务端的响应
        response = tcp_client.client_socket.recv(1024).decode("utf-8")
        print(f"服务端回复: {response}")

        time.sleep(1)
        

