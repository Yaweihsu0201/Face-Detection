#!/bin/bash

echo "script started"

mkdir -p SKinFilterResults_new
echo "mkdir done"

# 👉 指定你要的圖片
images=(
    "TestImagesForPrograms/31.jpg"
    "TestImagesForPrograms/33.jpg"
    "TestImagesForPrograms/mit3.jpg"
)

for img in "${images[@]}"
do
    [ -e "$img" ] || { echo "skip missing: $img"; continue; }

    filename=$(basename "$img")
    filename="${filename%.*}"

    out="SKinFilterResults_new/${filename}.png"

    python utilities/skin_filter.py "$img" "$out"

    echo "Finished processing $img"
done

echo "All images finished."