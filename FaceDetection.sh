#!/bin/bash

echo "script started"
mkdir -p FaceDetectionResults_new
echo "mkdir done"

for img in TestImagesForPrograms/*.jpg
do
    filename=$(basename "$img" .jpg)

    out="FaceDetectionResults_new/${filename}.png"

    python Face_detection_test.py "$img" "$out"

    echo "Finished processing $img"
done

echo "All images finished."