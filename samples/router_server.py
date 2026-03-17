import zmq
import time


def start_router():
    context = zmq.Context()
    # 创建 Router Socket
    # Router 是全能王，它能处理非阻塞的、多路并行的连接
    socket = context.socket(zmq.ROUTER)
    socket.bind("tcp://*:5556")

    print("Python Router 服务器已启动...")

    while True:
        try:
            # 接收多帧消息
            # Router 收到的消息结构通常是: [Identity, Empty, Data]
            frames = socket.recv_multipart()

            # 解析帧
            identity = frames[0]  # 第一帧是身份 ID
            empty_delimiter = frames[1]  # 第二帧通常是空帧（分隔符）
            message = frames[2]  # 第三帧是真实数据

            print(f"收到来自 [{identity.decode()}] 的消息: {message.decode()}")

            # 模拟异步处理（比如服务器在忙别的）
            # 在 Req-Rep 模式下，这里 sleep 会卡死所有客户端
            # 但在 Router 模式下，其他客户端的消息依然会进入队列，不会丢失
            time.sleep(0.1)

            # 发送回复
            # 关键规则：Router 发送时，必须把 Identity 放在第一帧！
            # 这样 ZMQ 才知道要把这两个字发给谁。
            response_data = b"Server Reply"

            print(f"正在回复 -> {identity.decode()}")

            # 原路返回：[Identity, Empty, Response]
            socket.send_multipart([identity, empty_delimiter, response_data])

        except KeyboardInterrupt:
            print("服务器停止")
            break


if __name__ == "__main__":
    start_router()
