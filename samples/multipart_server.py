import zmq
import time

def start_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("Python: 等待请求...")

    while True:
        # 1. 接收 Unity 的请求
        msg = socket.recv_string()
        print(f"收到: {msg}")

        # 2. 发送多帧回复
        # Frame 1: 指令 (String)
        # Frame 2: 数据 (Bytes)
        print("Python: 发送 LOAD_LEVEL 指令和数据...")
        
        # send_multipart 自动处理多帧发送
        # 注意：列表里即有 bytes 也有 string 转成的 bytes
        socket.send_multipart([
            b"LOAD_LEVEL", 
            b"\xAA\xBB\xCC\xDD" # 模拟的二进制数据
        ])

if __name__ == "__main__":
    start_server()