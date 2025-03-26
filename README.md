# image-mosaic
 Creates an image mosaic out of input files.

*Ensure dataset images are all the same size

## Usage
1. change `target_name` in mosaic.py to the name of the target image (configure folder and filetype in `target = cv2.imread()` on the line below)
2. change `width_tiles` to the width, in tiles (1 tile is 1 image from dataset)
3. change `folder` to the name of the folder in which the dataset of images is
4. run the file (`python3 mosaic.py`)