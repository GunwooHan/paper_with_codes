import parmap
import cv2
import numpy as np
import multiprocessing

print('smooth_target_data generating ...')
for filename in tqdm(filenames):
    img = cv2.imread(f'target_data/{filename}')
    img = make_edge_smooth(img)
    cv2.imwrite(f"smooth_data/{filename}",img)