# imports
import cv2
from anime_face_detector import create_detector
import glob
import os
import sys
from PIL import Image

# load model
detector = create_detector('yolov3')

# Padding rate
padding_left_rate = 0.5
padding_bottom_rate = 0.1
padding_right_rate = 0.5
padding_top_rate = 0.8

head_left   = 0
head_top    = 0
head_right  = 0
head_bottom = 0

face_width = 0
face_height = 0

def expand2square(pil_img, background_color):
  width, height = pil_img.size
  if width == height:
    return pil_img
  elif width > height:
    result = Image.new(pil_img.mode, (width, width), background_color)
    result.paste(pil_img, (0, (width - height) // 2))
    return result
  else:
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


# main
def getBody(image):

  height, width = image.shape[:2]

  body_left   = 0
  body_top    = (int)(head_bottom - (face_height * padding_bottom_rate))
  body_right  = width
  body_bottom = height

  sq_image = image[
      body_top : body_bottom,
      body_left : body_right
  ]

  return sq_image

def getHead(image):
  global face_width
  global face_height
  global head_left
  global head_top
  global head_righ
  global head_bottom

  preds = detector(image)

  if len(preds) == 0:
    return None

  left = preds[0]['bbox'][0]
  bottom = preds[0]['bbox'][1]
  right = preds[0]['bbox'][2]
  top = preds[0]['bbox'][3]
  face_x = int((left + right) / 2)
  face_y = int((top + bottom) / 2)
  height, width = image.shape[:2]

  face_width  = (int)(right - left)
  face_height = (int)(top - bottom)

  head_left   = (int)(left - (face_width * padding_left_rate))
  head_top    = (int)(bottom - (face_height * padding_top_rate))
  head_right  = (int)(right + (face_width * padding_right_rate))
  head_bottom = (int)(top + (face_height * padding_bottom_rate))

  if head_left < 0:
      head_left = 0
  if head_top < 0:
      head_top = 0
  if head_right > width:
      head_right = width
  if head_bottom > height:
      head_bottom = height

  sq_image = image[
      head_top : head_bottom,
      head_left : head_right
  ]

  return sq_image


args = sys.argv

print(args[1])
print(args[2])

input_dir = args[1]
output_dir = args[2]
# input_dir = r"C:/Macro/nk_out/face-body-splitter/input_data/"
# output_dir = r"C:/Macro/nk_out/face-body-splitter/output_data/"

output_extension = "png"#@param{type:"string"}

if not os.path.exists(output_dir):
  raise ValueError("output_dir is not exist")

paths = glob.glob(input_dir + "/*")
paths_len = len(paths)
error_list = []

for i, path in enumerate(paths):
  print(f"{i}/{paths_len} : {path}")
  basename = os.path.splitext(os.path.basename(path))[0]

  image = cv2.imread(path)
  head_image = getHead(image)
  body_image = getBody(image)

  # Skip if face does not exist
  if head_image is None:
    print("Could not recognize the face in this image")
    error_list.append(path)
    continue
  # Skip if face does not exist
  if body_image is None:
    print("Could not recognize the face in this image")
    error_list.append(path)
    continue

  cv2.imwrite(f"{output_dir}/head{i}.{output_extension}", head_image)
  im = Image.open(f'{output_dir}/head{i}.{output_extension}')
  im_new = expand2square(im, (0, 0, 0)).resize((512, 512))
  im_new.save(f'{output_dir}/head{i}.{output_extension}')

  cv2.imwrite(f"{output_dir}/body{i}.{output_extension}", body_image)
  im = Image.open(f'{output_dir}/body{i}.{output_extension}')
  im_new = expand2square(im, (0, 0, 0)).resize((512, 512))
  im_new.save(f'{output_dir}/body{i}.{output_extension}')

print("error list = ", error_list)
print("done. enjoy!")
