# CoCO-Splat (Compact CUDA OpenGL)

Efficient real-time renderer for the 3D Gaussian Splatting method based on [this](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) research paper.
Created for maximum efficiency with low VRAM demand. Written in C++17 with OpenGL 4.6 and CUDA Interop.

## Main optimizations
- **Pre-processing `.ply` files:** Stripping higher degrees of spherical harmonics and packing attributes into 28 B `FP16` structures inside `SSBO` buffers.
- **Opacity and scale pruning:** Combined with Opacity-Aware Tight AABB computation.
- **GPU sorting and culling (CUDA + CUB):** Includes multiple culling stages:
  - Near-plane culling
  - Frustum culling
  - Pixel culling
  - Stochastic LOD culling (index-based splat dropping using PCG32)
- **Three main modes:**
  - **Clean mode:** Pure rendering pipeline without extra overhead.
  - **Interactive mode:** Dear ImGui HUD, loading a camera from file, and built-in benchmarking.
  - **Screenshot mode:** Headless rendering of test camera views for metrics evaluation.

## Environment and build

```bash
# 1. Create and enter the Distrobox environment
distrobox assemble create --file distrobox.ini
distrobox enter <container-name>

# 2. Compile the project from the repository root
mkdir -p src/build && cd src/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_IMGUI=ON
cmake --build . -j$(nproc)
```
*In order to build without Dear ImGui, set `-DUSE_IMGUI=OFF`.*

## Starting up and environment variables

Start-up configuration is controlled via these environment variables:

| Variable | Default value | Comment |
| :--- | :--- | :--- |
| `RENDER_MODE` | `clean` | Available render modes: `clean`, `imgui`, `screenshot` |
| `PLY_FILE_PATH` | `../../DELL.ply` | File path to the `.ply` file with trained Gaussians |
| `TEST_CAMERAS_PATH` | `test_cameras.txt` | File path to the cameras exported by `render.py` from CoCO-Train |
| `SCREENSHOT_OUT_DIR` | `renders` | Output directory for rendered images |

### Start-up example

```bash
RENDER_MODE=imgui \
PLY_FILE_PATH=./models/scene.ply \
TEST_CAMERAS_PATH=./test_cameras/cameras.txt \
./src/build/CoCO-Splat
```

### Evaluation on the test subset (compatible with [CoCO-Train](https://github.com/RareIcubu/CoCO-Train))

```bash
./run_eval.sh /path/to/experiment
```
The script generates a full set of renders ready for metrics evaluation.

## Controls
- `W`/`A`/`S`/`D` – camera movement (`Left Shift` for 8x speed)
- Mouse / Scroll – camera rotation and FOV adjustment
- `TAB` – toggle cursor lock (for interacting with ImGui)
- `F1` – toggle GUI
- `F5` – switch between free camera and the first camera in `TEST_CAMERAS_PATH`
- `F10` – run 5-second benchmark (saves to `benchmark_<model>.txt`)
- `X` (hold) – wireframe preview (`GL_LINE` mode)
- `ESC` – close the app
