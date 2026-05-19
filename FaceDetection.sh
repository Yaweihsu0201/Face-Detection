#!/bin/bash

echo "script started"
mkdir -p FaceDetectionResults
for img in TestImagesForPrograms/*.jpg
do
    filename=$(basename "$img" .jpg)

    out="FaceDetectionResults/${filename}.png"

    python FaceDetection.py --input "$img" --output "$out"

    echo "Finished processing $img"
done

echo "All images finished." 