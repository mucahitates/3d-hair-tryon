# Hair Segmentation Module

This module handles hair removal from photos using AI models.

## Models
- BiSeNet (Face Parsing)
- MODNet Hair (Hair Matting)

## Usage
\\\python
python segment.py --input photos/ --output masks/
\\\

## Output
- face_mask.png (only face + skull + ears)
- hair_mask.png (only hair region)
