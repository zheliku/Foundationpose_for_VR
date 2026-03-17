import zmq
import time
import random


def start_weather_station():
    context = zmq.Context()

    # 1. 创建 Publisher Socket (PUB)
    # 这种 Socket 只能发，不能收
    publisher = context.socket(zmq.PUB)

    # 2. 绑定端口
    publisher.bind("tcp://*:5556")
    print("Python 气象站已启动 (端口 5556)...")

    # 注意：防止“慢订阅者”问题 (Slow Joiner)
    # PUB 套接字在绑定后，如果立刻发送消息，而此时还没有人连上来，
    # 消息会直接丢弃！就像对着空房间说话。
    # 所以我们通常稍微等一下，或者在循环里一直发。
    time.sleep(1)

    while True:
        # 随机生成一些数据
        temperature = random.randint(20, 35)
        humidity = random.randint(40, 80)

        # --- 发送第一类消息：Weather ---
        topic_weather = b"Weather"
        data_weather = f"Temp: {temperature}C, Humidity: {humidity}%".encode("utf-8")

        print(f"广播: [Weather] {data_weather}")
        # 发送多帧：[Topic, Data]
        publisher.send_multipart([topic_weather, data_weather])

        # --- 发送第二类消息：Alert (模拟偶尔发生) ---
        if random.random() > 0.7:
            topic_alert = b"Alert"
            data_alert = b"Typhoon incoming!"
            print(f"广播: [Alert] {data_alert}")
            publisher.send_multipart([topic_alert, data_alert])

        time.sleep(1)


if __name__ == "__main__":
    start_weather_station()
