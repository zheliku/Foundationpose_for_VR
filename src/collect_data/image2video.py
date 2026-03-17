import os
import re
import glob
import cv2


def _natural_key(s: str):
    base = os.path.basename(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]


def images_to_mp4(input_dir: str, output_path: str, fps: int = 30) -> None:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(input_dir, ext)))

    if not imgs:
        print(f"未找到图片：{input_dir}")
        return

    imgs.sort(key=_natural_key)

    first = cv2.imread(imgs[0])
    if first is None:
        print(f"无法读取首帧：{imgs[0]}")
        return

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 写入首帧
    writer.write(first)

    for p in imgs[1:]:
        img = cv2.imread(p)
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        writer.write(img)

    writer.release()
    print(f"已保存：{output_path}")


if __name__ == "__main__":
    input_dir = r"test_data/cube/output/pose_visualization"
    output_path = os.path.join(os.path.dirname(input_dir), "output.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images_to_mp4(input_dir, output_path, fps=30)