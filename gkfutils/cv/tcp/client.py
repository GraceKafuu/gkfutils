import socket

if __name__ == "__main__":
    # 创建 TCP 客户端
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 连接服务端
    server_address = ("127.0.0.1", 12345)
    client_socket.connect(server_address)
    print(f"服务端 {server_address} 连接成功！")

    while True:
        # 发送数据到服务端
        message = input("请输入要发送的消息: ")
        client_socket.send(message.encode("utf-8"))
        if not message:
            break

        # 接收服务端的响应
        response = client_socket.recv(1024).decode("utf-8")
        print(f"服务端回复: {response}")

    # 关闭连接
    client_socket.close()
