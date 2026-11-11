#!/usr/bin/env bash
# Run this task once and put the videos in your home folder with plain names.
#
#   bash run.sh          the scripted demo  -> ~/<task>_demo_top.mp4, _demo_left.mp4, _demo_right.mp4, ~/<task>_setup.png
#   bash run.sh dream    Dream-1            -> ~/<task>_dream_top.mp4, _dream_left.mp4, _dream_right.mp4
#
# Prints the score at the end. The demo records at 640 x 360; Dream-1 at 320 x 176, the size it is built for.
set -u
source /home/ubuntu/IsaacLab-Arena/yam.env 2>/dev/null || source ~/IsaacLab-Arena/yam.env
HERE=$(cd "$(dirname "$0")" && pwd)
TASK=$(basename "$HERE")
MODE=${1:-demo}
if [ "$MODE" = dream ]; then POLICY=dream; STEPS=1200; W=320; H=176; else POLICY=yam_scripted; STEPS=600; W=640; H=360; fi
LOG=~/yam-eval/$TASK-$MODE
rm -rf "$LOG" "$LOG.txt"
cd "$HERE"
INSTRUCTION=$(python -c "import json; print(json.load(open('task.json'))['instruction'])")
echo "Running $TASK ($MODE). This takes a few minutes; the screen stays quiet until it is done."
yes skip | inspect-robots run --embodiment yam_arena --policy $POLICY \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=$W -E cam_height=$H \
  --instruction "$INSTRUCTION" --scorer task_score --max-steps $STEPS --store-frames --log-dir "$LOG" \
  > "$LOG.txt" 2>&1
for log in "$LOG"/*.json; do inspect-robots video "$log" > /dev/null 2>&1; done
for cam in top left right; do
  src=$(find "$LOG/frames" -name "*${cam}_cam.mp4" | head -1)
  [ -n "$src" ] && ffmpeg -loglevel error -y -i "$src" -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
    ~/"${TASK}_${MODE}_${cam}.mp4"
done
[ "$MODE" = demo ] && ffmpeg -loglevel error -y -i ~/"${TASK}_demo_top.mp4" -frames:v 1 ~/"${TASK}_setup.png"
echo
echo "Score: $(grep -o -i 'task_score[^,}]*' "$LOG.txt" | tail -1)"
grep -i -E "Traceback|Error" "$LOG.txt" | tail -3
echo "Videos in your home folder:"
ls ~/"${TASK}_${MODE}"_*.mp4 2>/dev/null
