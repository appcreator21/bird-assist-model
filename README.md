# Bird Assist model (release assets)

Runtime files for running Bird Assist directly, without the website: the half-precision ONNX model
(fine-tuned DINOv2 ViT-B/14, 829 US species), class labels, the location/season range table, and
`identify.py`, which reproduces the site's preprocessing, temperature (0.75), range filter and % display.

Download the assets from the latest release, then: `python3 identify.py photo.jpg --assets <dir>`.
The hosted app: https://matthuber-bird-assist.static.hf.space/
