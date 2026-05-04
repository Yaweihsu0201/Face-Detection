#!/bin/bash

echo "script started"
mkdir -p FaceDetectionResult_single
echo "mkdir done"


img="TestImagesForPrograms/19.jpg"

out="FaceDetectionResult_single/19.png"

python Face_detection_test.py "$img" "$out"

echo "Finished processing $img"

echo "All images finished."