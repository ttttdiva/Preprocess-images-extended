import os
import argparse

import cv2
from PIL import Image
import numpy as np



# 元画像パス受け取り
parser = argparse.ArgumentParser()
parser.add_argument("--src_dir")
parser.add_argument("--out_dir")
args = parser.parse_args()

# src_dir = r"C:/Macro/nk_out/WhiteBG/input_data/"
in_files = os.listdir(args.src_dir)

# 出力先パス
# out_dir = r"input_data/"

for i in range(len(in_files)):
    # 白抜き後のファイルパス
    in_dfiles = args.src_dir + in_files[i]
    out_dfiles = args.out_dir + in_files[i]
    print(out_dfiles)

    # 画像を読み込んでNumPy配列を作成
    image_array = np.array(Image.open(in_dfiles))
    # image_array = cv2.imread(IMAGE_PATH, -1)

    B, G, R, A = cv2.split(image_array)
    alpha = A / 255

    R = (255 * (1 - alpha) + R * alpha).astype(np.uint8)
    G = (255 * (1 - alpha) + G * alpha).astype(np.uint8)
    B = (255 * (1 - alpha) + B * alpha).astype(np.uint8)

    image = cv2.merge((B, G, R))

    # アルファチャンネルのみの画像を作成して保存
    alpha_image = Image.fromarray(image)
    alpha_image.save(out_dfiles)
