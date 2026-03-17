import zmq
import time


def start_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("Python Server: 启动成功，等待多帧消息...")

    while True:
        try:
            # 1. 接收多帧消息
            # recv_multipart 返回一个列表，里面是 bytes 类型的帧
            # 例如: [b'LOGIN', b'Zelda', b'123456']
            frames = socket.recv_multipart()

            print(f"\n收到消息，共 {len(frames)} 帧")

            # 解析第一帧（通常是指令）
            # decode() 是将 bytes 转为 string
            command = frames[0].decode("utf-8")

            if command == "LOGIN":
                # 确保帧数量正确，防止索引越界
                if len(frames) >= 3:
                    username = frames[1].decode("utf-8")
                    password = frames[2].decode("utf-8")
                    print(f"处理登录请求 -> 用户: {username}, 密码: {password}")

                    # 模拟验证过程
                    time.sleep(0.5)

                    # 2. 发送多帧回复
                    # send_multipart 接受一个列表
                    print("发送回复: [STATUS, OK]")
                    socket.send_multipart([b"STATUS", b"OK"])
                else:
                    print("错误: LOGIN 指令缺少参数")
                    socket.send_multipart([b"STATUS", b"ERROR_MISSING_ARGS"])

            else:
                print(f"未知指令: {command}")
                socket.send_multipart([b"STATUS", b"UNKNOWN_CMD"])

        except KeyboardInterrupt:
            print("服务器停止")
            break


if __name__ == "__main__":
    start_server()
