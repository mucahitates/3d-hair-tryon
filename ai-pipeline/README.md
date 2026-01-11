# AI Pipeline

This module contains the full AI and 3D processing pipeline used to convert user photos into a high-quality 3D head model and apply selected hairstyles.

The pipeline is executed by GPU workers and consists of multiple independent stages:

1. Photo validation
2. Hair segmentation
3. Photogrammetry (Metashape)
4. Mesh cleanup and optimization
5. Hair fitting and deformation
6. Quality control and retry logic

Each stage is designed to be modular, reproducible, and GPU-accelerated.
