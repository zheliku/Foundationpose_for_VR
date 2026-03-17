import time
import zmq

def start_server():
    # 1. 创建 Context（上下文），这是 ZMQ 的全局环境
    context = zmq.Context()

    # 2. 创建 Socket，类型为 REP (Reply/Response)
    # 对应文档中的：server = new ResponseSocket()
    socket = context.socket(zmq.REP)

    # 3. 绑定端口 (Bind)
    # 我们绑定到 5555 端口。
    # "tcp://*:5555" 意味着监听本机所有网卡的 5555 端口
    socket.bind("tcp://*:5555")
    print("Python 服务端已启动，正在监听端口 5555...")

    while True:
        # 4. 接收消息 (Receive)
        # recv_string 会阻塞在这里，直到收到消息为止
        # 对应文档：server.ReceiveFrameString()
        message = socket.recv_string()
        print(f"收到请求: {message}")

        # 模拟一些处理时间，比如服务器在计算
        time.sleep(0.1)  # 休息 100 毫秒

        # 5. 发送回复 (Send)
        # 对应文档：server.SendFrame("World")
        response = "World"
        print(f"发送回复: {response}")
        socket.send_string(response)

if __name__ == "__main__":
    start_server()