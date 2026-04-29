#!/bin/bash

echo "script started"
mkdir -p FaceDetectionResult_single
echo "mkdir done"


img="TestImagesForPrograms/mit3.jpg"

out="FaceDetectionResult_single/mit3.png"

python Face_detection_test.py "$img" "$out"

echo "Finished processing $img"

echo "All images finished."