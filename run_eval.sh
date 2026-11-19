#!/usr/bin/env bash

EXP_DIR="$1"

if [ -z "$EXP_DIR" ]; then
    echo "Usage: $0 <path to gaussian splatting processed directory>"
    exit 1
fi

export RENDER_MODE="screenshot"
export PLY_FILE_PATH="$EXP_DIR/point_cloud/iteration_30000/point_cloud.ply"
export TEST_CAMERAS_PATH="$EXP_DIR/test_cameras.txt"
export SCREENSHOT_OUT_DIR="$EXP_DIR/test/coco_splat_30000/renders"

mkdir -p "$SCREENSHOT_OUT_DIR"
ln -sfn "$(realpath "$EXP_DIR/test/ours_30000/gt")" "$EXP_DIR/test/coco_splat_30000/gt"

./src/build/CoCO-Splat
