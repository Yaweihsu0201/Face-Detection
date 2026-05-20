# Face Detection with Skin Segmentation and Facial Landmark Triplet Scoring

This project implements a face detection pipeline inspired by the method proposed in:

> Ke-Jie Liao 廖科傑, "運用輪廓色彩和五官的人臉偵測技術 Face Detection by Outline, Color, and Facial Features", 2010.

This project implements a face detection pipeline based on skin segmentation, ellipse matching, facial feature maps, geometric filtering, and machine learning-based triplet scoring.

Compared with the original method, this implementation includes several modifications, including:

- Replace color-based skin filtering with Deep Learning method
- Add several more eye-mouth features into consideration
- Select the most likely facial landmark triplet using geometric constraints and a trained Random Forest classifier.

## Overview

The goal of this project is to detect faces in an image by identifying the best combination of:

- left eye
- right eye
- mouth

within candidate face ellipses.

The final output visualizes the detected facial structure by drawing:

- the fitted face ellipse
- detected eye and mouth points
- lines connecting the eyes and mouth

## Method

The detection pipeline consists of the following steps:

1. **Skin Segmentation**  
   A trained U-Net model is used to predict skin regions from the input image.

2. **Ellipse Matching**  
   Face-like elliptical regions are extracted from the predicted skin mask.

3. **Eye Map and Mouth Map Generation**  
   Eye and mouth response maps are generated from the RGB image.

4. **Candidate Extraction**  
   Local maxima from the eye map and mouth map are selected as eye and mouth candidates.

5. **Geometric Filtering**  
   Candidate triplets are filtered using facial geometry rules, such as:
   - the two eyes should have a reasonable distance
   - the eye midpoint should lie close to the face major axis
   - the mouth should be roughly aligned with the face major axis
   - the eye line should be approximately perpendicular to the eye-mouth direction

6. **Triplet Scoring**  
   A Random Forest classifier scores each valid triplet based on normalized geometric features.

7. **Best Triplet Selection**  
   The highest-scoring triplet across all candidate ellipses is selected as the final detection result.

## Project Structure

```text
Face-Detection/
│
├── main.py
├── detector.py
│
├── utilities/
│   ├── predict_skin.py
│   ├── ellipse_matching.py
│   ├── eyemap.py
│   ├── mouthmap.py
│   ├── triplet_finder.py
│   ├── triplet_scorer.py
│   ├── triplet_geometry.py
│   └── visualization.py
│
├── checkpoints/
│   ├── unet_skin_best.pth
│   └── triplet_scorer_rf.pkl
│
├── TestImagesForPrograms/
├── FaceDetectionResults/
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Yaweihsu0201/Rule-Based-Face-Detection.git
cd Face-Detection
```

Create a virtual environment:

```bash
py -3.10 -m venv .venv
```

Activate the virtual environment:

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run face detection on a single image:

```bash
python main.py --input TestImagesForPrograms/23.jpg --output FaceDetectionResults/23.png
```

Arguments:

```text
--input     Path to the input image
--output    Path to save the detection result
```

Example:

```bash
python main.py --input TestImagesForPrograms/001.jpg --output FaceDetectionResults/001.png
```

## Results example

| Original Image | Detection Result |
|---------------|----------------|
| ![](docs/19.jpg) | ![](docs/19.png) |

## Pipeline Details

### Skin Segmentation

The skin segmentation model predicts a binary or probability mask of skin regions. This mask is used to locate possible face areas.

### Ellipse Matching

Connected skin regions are analyzed and approximated using ellipses. Each ellipse is treated as a possible face candidate.

### Candidate Detection

Eye and mouth response maps are computed from the input image. For each ellipse, local maxima inside the ellipse are selected as possible facial feature candidates.

### Geometric Triplet Filtering

For each ellipse, combinations of two eye candidates and one mouth candidate are tested using geometry-based constraints.

The geometry module checks whether the triplet forms a reasonable facial structure.

### Random Forest Triplet Scoring

After geometric filtering, each valid triplet is converted into a feature vector. These features include:

- normalized eye-to-eye distance
- normalized eye-to-mouth distance
- mouth-to-eye-midpoint distance
- relative coordinates inside the ellipse
- angles between facial axes
- ellipse aspect ratio
- eye map and mouth map response values

A trained Random Forest classifier predicts the confidence score of each triplet. The best-scoring triplet is selected as the final result.

## Future Improvements

Possible future improvements include:

- Support batch processing for an entire image folder
- Add evaluation metrics for face detection accuracy
- Improve candidate extraction efficiency
- Replace manual thresholding with adaptive thresholds
- Train a stronger triplet scoring model
- Compare performance with modern face detectors such as MTCNN, RetinaFace, or YOLO-based detectors
- Add support for multiple face detection in one image
