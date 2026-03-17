import zmq
import time


def start_server():
    context = zmq.Context()

    # 1. 创建 Response Socket (REP)
    # 它是被动的，必须先收，再发
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("Python REP 服务端已启动...")

    while True:
        try:
            # 2. 【必须】先接收请求 (Receive)
            # 如果没有请求发过来，代码会卡在这里死等
            message = socket.recv_string()
            print(f"收到客户端请求: {message}")

            # 模拟思考时间
            time.sleep(0.5)

            # 3. 【必须】发送回应 (Send)
            # 如果你这里忘了 send，或者试图连续 recv 两次，
            # 下次循环时就会报错 (EFSM 错误)
            reply = "Who's there?"
            print(f"发送回应: {reply}")
            socket.send_string(reply)

            # 循环回到开头，再次进入 Receive 状态

        except zmq.ZMQError as e:
            print(f"ZMQ 错误: {e}")
            break
        except KeyboardInterrupt:
            print("退出")
            break


if __name__ == "__main__":
    start_server()
