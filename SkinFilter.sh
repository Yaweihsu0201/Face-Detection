#!/bin/bash

echo "script started"

mkdir -p SKinFilterResults_new
echo "mkdir done"

for img in TestImagesForPrograms/*.{jpg,JPG,jpeg,JPEG}
do
    [ -e "$img" ] || continue  

    filename=$(basename "$img")
    filename="${filename%.*}"  

    out="SKinFilterResults_new/${filename}.png"

    python utilities/skin_filter.py "$img" "$out"

    echo "Finished processing $img"
done

echo "All images finished.