from PencilDrawingBySketchAndTone import *
import matplotlib.pyplot as plt
import torchvision
import time
import os
from tqdm import tqdm


load_folder = '/home/math/oberman-lab/datasets/imagenet/val'
load_folder = '/mnt/data/scratch/ImageNet/val'
imagenet = torchvision.datasets.ImageFolder(load_folder, transform=torchvision.transforms.ToTensor())


pencils = {
#     'pencil0':'pencils/pencil0.jpg',
    'pencil1':'pencils/pencil1.jpg',
    'pencil2':'pencils/pencil2.png',
    'pencil3':'pencils/pencil3.jpg',
#     'pencil4':'pencils/pencil4.jpg'
}

i = 0
for root, dirs, files in tqdm(os.walk(load_folder, topdown=False), desc='Folders', total=len(os.listdir(load_folder))):
    for pencil in tqdm(pencils, desc='Pencils', leave=False):
        pencil_tex = pencils[pencil]
#         save_folder = f'../imagenet-color-sketch-{pencil}/val'
        save_folder = f'/mnt/data/scratch/tiago.salvador/imagenet-color-sketch-{pencil}/val'
        if not os.path.exists(root.replace(load_folder, save_folder)):
            os.makedirs(root.replace(load_folder, save_folder), exist_ok=True)
        for name in tqdm(files[40:], desc='Images', leave=False):
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


