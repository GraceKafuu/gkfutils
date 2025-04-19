import zmq
import threading


class ZMQ_Client(object):
    def __init__(self, port1: int=5555, port2: int=5556):
        self.port1 = port1
        self.port2 = port2
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        # 连接到两个服务端
        self.server1 = "tcp://localhost:{}".format(self.port1)
        self.server2 = "tcp://localhost:{}".format(self.port2)
        self.socket.connect(self.server1)
        self.socket.connect(self.server2)
        print("客户端已启动, 已连接到服务端：{} {}".format(self.server1, self.server2))

        # 订阅所有消息（空字符串表示不过滤）
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def run(self):
        print("等待接收消息...\n")
        while True:
            message = self.socket.recv_string()
            print(f"已收到消息: {message}")

    def start_run(self):
        try:
            t = threading.Thread(target=self.run, args=())
            t.start()
        except Exception as e:
            print("Error: 无法启动线程 {}".format(e))


if __name__ == "__main__":
    server = ZMQ_Client(port1=5555, port2=5556)
    server.run()


