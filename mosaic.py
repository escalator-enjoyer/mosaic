import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# loading images
def load_images(folder):
    images = []
    colors = []
    
    files = os.listdir(folder)
    random.shuffle(files)
    files = files[::1]

    with tqdm(total=len(files)) as pb:
        for filename in files:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                avg_color = np.mean(lab_img, axis=(0, 1))
                images.append(img)
                colors.append(avg_color)
                pb.update(1)
    
    return images, np.array(colors)

# john pythagoras if he was actually vector pythagoras
def color_distance(c1, c2):
    return np.linalg.norm(c1 - c2)

# generating the actual mosaic
def generate_mosaic(target_image, images, colors, output_path, width_in_tiles, chunk_size=8):
    h, w, _ = target_image.shape
    img_h, img_w, _ = images[0].shape

    height_in_tiles = int(h * width_in_tiles / w)
    output_h, output_w = height_in_tiles * img_h, width_in_tiles * img_w

    lab_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB)

    # select from top 1%
    top_n = max(int(0.01 * len(images)), 1)
    
    mosaic_rows = []
    with tqdm(total=width_in_tiles * height_in_tiles) as pb:
        for y in range(0, height_in_tiles, chunk_size):
            rows_in_chunk = min(chunk_size, height_in_tiles - y)
            temp_chunk = np.zeros((rows_in_chunk * img_h, output_w, 3), dtype=np.uint8)

            for chunk_y in range(rows_in_chunk):
                for x in range(width_in_tiles):
                    pixel_color = lab_target[int((y + chunk_y) * h / height_in_tiles),
                                            int(x * w / width_in_tiles)]
                    distances = np.array([color_distance(pixel_color, c) for c in colors])
                    sorted_indices = np.argsort(distances)
                    
                    chosen_img = images[random.choice(sorted_indices[:top_n])]
                    chosen_img_resized = cv2.resize(chosen_img, (img_w, img_h))

                    y1, y2 = chunk_y * img_h, (chunk_y + 1) * img_h
                    x1, x2 = x * img_w, (x + 1) * img_w
                    
                    pb.update(1)
                    temp_chunk[y1:y2, x1:x2] = chosen_img_resized
            
            mosaic_rows.append(temp_chunk)
    
    final_mosaic = cv2.vconcat(mosaic_rows)
    cv2.imwrite(output_path, final_mosaic)
    return output_path

target_name = "roingus"
target = cv2.imread(f"io/input/{target_name}.png")
# 33% stretch of original card because pokemon card aspect ratio
# target = cv2.resize(target, (int(target.shape[1] * 1.33), target.shape[0]))
width_tiles = 32

print("loading images")
folder = "album_covers"
images, colors = load_images(folder)
print("loading done")

print("generating mosaic")
output_path = f"io/output/{folder}_{target_name}_{width_tiles}w.png"
generate_mosaic(target, images, colors, output_path, width_tiles)
print("mosaic done")
