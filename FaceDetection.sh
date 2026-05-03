#!/bin/bash

echo "script started"
mkdir -p FaceDetectionResults
echo "mkdir done"

for img in TestImagesForPrograms/*.jpg
do
    filename=$(basename "$img" .jpg)

    out="FaceDetectionResults/${filename}.png"

    python Face_detection_test.py "$img" "$out"

    echo "Finished processing $img"
done

echo "All images finished."