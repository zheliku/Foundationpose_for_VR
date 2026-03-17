# pgm_server.py
import zmq
import time


def start_broadcaster():
    context = zmq.Context()
    # 使用 PUB (发布者) 模式
    socket = context.socket(zmq.PUB)

    # PGM 设置 (根据文档建议)
    # Rate: 速率限制 (40Mbps)
    socket.setsockopt(zmq.RATE, 40 * 1024 * 1024)
    # Recovery IVL: 恢复间隔
    socket.setsockopt(zmq.RECOVERY_IVL, 1000 * 60 * 10)  # 10分钟

    # 绑定 PGM 地址
    # 注意格式: pgm://网卡接口;组播IP:端口
    # 在 Python pyzmq 中，pgm 的地址格式通常比较严格
    # pgm://interface;multicast_group:port
    # 127.0.0.1 可能在某些系统不支持 PGM，建议换成局域网真实 IP，或者尝试 epqm (OpenPGM)
    # 这里为了演示尝试使用通用格式，如果报错，请换成 tcp 测试，因为 PGM 强依赖网络环境
    try:
        # epgm 是 Encapsulated PGM，pyzmq 常用这个
        socket.bind("pgm://127.0.0.1:5555")
        print("Python 广播电台已启动 (EPGM)...")
    except zmq.ZMQError as e:
        print(f"PGM 绑定失败 (可能是权限或不支持): {e}")
        print("切换回 TCP 模式演示 PUB/SUB")
        socket.bind("tcp://*:5555")

    count = 0
    while True:
        msg = f"广播消息 #{count}"
        print(f"正在广播: {msg}")

        # Topic(主题) + 内容。这里我们不设主题，直接发内容
        socket.send_string(msg)

        count += 1
        time.sleep(1)


if __name__ == "__main__":
    start_broadcaster()
