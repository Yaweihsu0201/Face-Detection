#!/bin/bash

echo "script started"
mkdir -p EllpiseMatchingResults1
echo "mkdir done"

i=mit3

img="SKinFilterResults_new/${i}.png"
clr="TestImagesForPrograms/${i}.jpg"
out="EllpiseMatchingResults1/${i}.png"

python scripts/EllipseMatching.py "$img" "$clr" "$out"

echo "Finished processing $img"
echo "All images finished."