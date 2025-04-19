import zmq
import time
import sys
 


class ZMQ_Server(object):
   def __init__(self, port: int=5555):
      self.port = port
      self.context = zmq.Context()
      self.socket = self.context.socket(zmq.PUB)
      self.client = "tcp://*:{}".format(self.port)
      self.socket.bind(self.client)
      print("服务端已启动, 已绑定客户端：{}".format(self.client))

   def send_msg(self, msg: str):
      self.socket.send(msg.encode('utf-8'))
      print("服务端已发送：{}".format(msg))


if __name__ == "__main__":
   client = ZMQ_Server(port=5555)
   while True:
      client.send_msg("Hello 5555")
      time.sleep(1)