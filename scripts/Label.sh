#!/bin/bash

echo "script started"

for img in dataset/*.jpeg
do
    filename=$(basename "$img" .jpeg)
    python Labeling.py "$img"

    echo "Finished processing $img"
done

echo "All images finished."