import trimesh
from pathlib import Path

def test_trimesh_load():
    obj_path = Path(__file__).parent / 'data' / 'Cube_Aruco.obj'
    print(f'Loading mesh from: {obj_path}')
    mesh: trimesh.Trimesh = trimesh.load(obj_path)
    print(mesh)
    mesh.show()

test_trimesh_load()