from PencilDrawingBySketchAndTone import *
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import time
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser(description='Create ImageNet-Drawing')
parser.add_argument('--model_path', type=str, default='saved_models', help="Location of the pre-trained models")
parser.add_argument('--load_folder', type=str, default='path/to/imagenet/val', help="Path to ImageNet val folder")
parser.add_argument('--save_folder', type=str, default='../datasets/imagenet-drawing', help="Path where to save ImageNet-Drawing") 
parser.add_argument('--drawing_pattern_path', type=str, default='drawing-patterns/drawing-pattern-I.jpg', help="Path to drawing pattern image") 
args = parser.parse_args()

def convert_image(name, root, load_folder, save_folder, pencil_tex):
    load_path = os.path.join(root, name)
    save_path = load_path.replace(load_folder, save_folder)
    if not os.path.exists(save_path):  
        img = cv2.cvtColor(cv2.imread(load_path), cv2.COLOR_BGR2RGB)
        im_pen = gen_pencil_drawing(img, kernel_size=8, stroke_width=1, num_of_directions=8, smooth_kernel="gauss",
                   gradient_method=1, rgb=True, w_group=2, pencil_texture_path=pencil_tex,
                           stroke_darkness=2, tone_darkness=1.5)
        if np.isnan(im_pen).any():
            tqdm.write(f'Failed {pencil} {load_path}')
        im_pen = exposure.rescale_intensity(im_pen, in_range=(0,1))
        cv2.imwrite(save_path, cv2.cvtColor(np.clip(im_pen*255, 0, 255).astype(np.uint8),cv2.COLOR_RGB2BGR))

for root, dirs, files in tqdm(os.walk(args.load_folder, topdown=False), desc='Folders', total=len(os.listdir(args.load_folder))):
    if not os.path.exists(root.replace(args.load_folder, args.save_folder)):
        os.makedirs(root.replace(args.load_folder, args.save_folder), exist_ok=True)
    Parallel(n_jobs=10)(delayed(convert_image)(files[i], root, args.load_folder, args.save_folder, args.drawing_pattern_path) for i in tqdm(range(len(files)), desc='Images', leave=False))