#!/bin/bash
if [ -z "$BLENDER" ]; then
    export BLENDER="blender"
fi

"$BLENDER" --background --python /home/lch/Downloads/bpy-visualization-utils-master/render_txt.py  --txt /home/lch/Downloads/bpy-visualization-utils-master/examples/0.txt -- --output /home/lch/Downloads/bpy-visualization-utils-master/examples/0.png