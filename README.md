# Simple Object Detection with CLIP

This project demonstrates a basic method for performing object detection using OpenAI's CLIP model. 
The goal is to detect cars in an image by adapting a model that understands the relationship between images and text. CLIP is designed to determine how well a text description matches an image. This project implements a **sliding window** approach.

The core idea is to:
1. Scan an image with rectangles of three sizes.
2. For each window (patch), ask the CLIP model: "How well does this patch match the description 'a photo of a car'?"
3. Windows with a high similarity score are considered potential detections.
4. Apply **Non-Maximum Suppression (NMS)** with **Intersection over Union** to merge overlapping boxes.

## Process

The notebook walks through the entire process:

1. **Setup**: Installs and imports necessary libraries.
2. **Model Loading**: Loads the pre-trained `ViT-B/16` CLIP model and its preprocessor.
3. **Image and Text Preparation**:
    - Loads the target image from `image_path`.
    - Defines text prompts for classification: `["car", "not a car"]`.
4. **Sliding Window Detection**:
    - Iterates over the image with multiple window sizes (`64x64`, `128x128`, `192x192`).
    - For each window, it crops the image and calculates the cosine similarity between the image patch and the text prompts.
    - If the confidence score for "car" is above a set threshold (we use`0.90`), the window's coordinates are stored as a potential detection.
5. **Non-Maximum Suppression (NMS)**:
    - The numerous overlapping detections are filtered using NMS.
    - This step cleans up the results.
6. **Visualization**:
    - The final bounding boxes are drawn on the original image.
    - The result is saved and displayed.


### This is a demonstration of the concept. Efficiency, speed and accuracy are not optimized.
