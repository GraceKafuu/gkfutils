import socket


if __name__ == "__main__":
    # 创建 TCP 服务端
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定 IP 地址和端口
    server_address = ("127.0.0.1", 12345)  # 绑定本机 IP 和端口 12345
    server_socket.bind(server_address)

    # 监听客户端连接
    server_socket.listen(5)  # 最大连接数为 5
    print(f"服务端 {server_address} 已启动, 等待客户端连接...")

    while True:
        # 接受客户端连接
        client_socket, client_address = server_socket.accept()
        print(f"客户端 {client_address} 连接成功！")

        while True:
            # 接收客户端发送的数据
            data = client_socket.recv(1024).decode("utf-8")
            if not data:
                break
            print(f"收到客户端消息: {data}")

            # 发送数据给客户端
            response = f"服务端已收到你的消息: {data}"
            client_socket.send(response.encode("utf-8"))

        # 关闭连接
        client_socket.close()
        server_socket.close()


