#!/bin/bash

echo "script started"
mkdir -p EllpiseMatchingResults1
echo "mkdir done"

for i in {11..20}
do
    img="result/${i}.png"
    clr="TestImagesForPrograms/${i}.jpg"
    out="EllpiseMatchingResults1/${i}.png"

    python scripts/EllipseMatching.py "$img" "$clr" "$out"

    echo "Finished processing $img"
done

echo "All images finished."