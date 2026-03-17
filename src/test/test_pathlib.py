from pathlib import Path

import numpy as np
from numpy import array, dtype, float32

p = Path("test_data/cube/output/pose_visualization")
print(p.parent)
print(p.name)
print(p.suffix)
print(p.stem)
print(list(p.glob("*.png")))
print(p.exists())
print(p.is_dir())
print(p.is_file())
print(p.resolve())

p2 = Path("test_data/cube/output/pose_visualization/2")
p2.mkdir(parents=True, exist_ok=True)

print(not [])

