#!/bin/bash

echo "script started"

img="TestImagesForPrograms/23.jpg"

filename=$(basename "$img" .jpg)
python Labeling.py "$img"

echo "Finished processing $img"

echo "All images finished."