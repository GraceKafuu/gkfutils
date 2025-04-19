import zmq
import threading


class ZMQ_Client(object):
    def __init__(self, port: int=5555):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        self.server = "tcp://localhost:{}".format(self.port)
        self.socket.connect(self.server)
        self.socket.setsockopt(zmq.SUBSCRIBE, ''.encode('utf-8'))
        print("客户端已启动, 已连接到服务端：{}".format(self.server))

    def run(self):
        print("等待接收消息...\n")
        while True:
            response = self.socket.recv().decode('utf-8')
            print(f"已收到消息: {response}")

    def start_run(self):
        try:
            t = threading.Thread(target=self.run, args=())
            t.start()
        except Exception as e:
            print("Error: 无法启动线程 {}".format(e))


if __name__ == '__main__':
    client = ZMQ_Client(port=5556)
    client.run()