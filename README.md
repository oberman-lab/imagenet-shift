# Shift-Happens

### ImageNet-Cartoon

##### Installation

Assuming you already have NVIDIA GPU and CUDA CuDNN installed, you can create a conda environment with all the requirements using the following commands:

```
conda create --name cartoon tensorflow-gpu==1.12.0
conda activate create
conda install -c conda-forge scikit-image==0.14.2
conda install -c conda-forge opencv
conda install -c conda-forge tqdm
```

##### Run Code

```
conda activate cartoon
cd src-imagenet-cartoon
python cartoonize.py --load_folder path/to/imagenet/val --save_folder ../datasets/imagenet-cartoon
```

Using a single Tesla P100 PCIe with 16 GB, generating ImageNet-Cartoon should take under 48 minutes.

### ImageNet-Drawing

##### Installation

You can create a conda environment with all the requirements using the following command:

```
conda create --name drawing -c conda-forge time matplotlib scikit-learn tqdm scikit-image python==3.9 opencv
```

##### Run Code

```
conda activate drawing
cd src-imagenet-drawing
python drawing.py --load_folder path/to/imagenet/val --save_folder ../datasets/imagenet-drawing
```

Unlike ImageNet-Cartoon, ImageNet-Drawing will take longer to be generated, around 8 hours.

To generate ImageNet-Drawing-II, ImageNet-Drawing-III and ImageNet-Drawing-IV, use the following:

```
python drawing.py --load_folder path/to/imagenet/val --save_folder ../datasets/imagenet-drawing-II --drawing-pattern drawing-patterns/drawing-pattern-II.jpg
python drawing.py --load_folder path/to/imagenet/val --save_folder ../datasets/imagenet-drawing-III --drawing-pattern drawing-patterns/drawing-pattern-III.png
python drawing.py --load_folder path/to/imagenet/val --save_folder ../datasets/imagenet-drawing-IV --drawing-pattern drawing-patterns/drawing-pattern-IV.jpg
```

To visualize the drawing process use the Visualization Drawing Jupyter Notebook included in the *src-imagenet-drawing*. This requires installing Jupyter Notebook on the drawing environment which can be accomplished with following command:

```
conda install -c conda-forge jupyterlab
```

### Results

##### Installation

You can create a conda environment with all the requirements using the following command:

```
conda create --name results pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda activate results
conda install -c conda-forge time
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
conda install -c conda-forge jupyterlab prettytable ipywidgets tqdm
conda install pandas
conda install -c conda-forge opencv
```

##### Run Code

```
python compute_metrics.py
```

Then use the "Compute Metrics" Jupyter Notebook to generate the Tables in the paper. With the Visualization Jupyter Notebook, one can display the different images of ImageNet-Cartoon and ImageNer-Drawing in the paper.

### References

The code in the folders src-imagenet-cartoon and src-imagenet-drawing are taken from [1] and [2], respectively, with minimum changes. 

[1] Tensorflow implementation for CVPR2020 paper “Learning to Cartoonize Using White-box Cartoon Representations”. [GitHub Repo Link](https://github.com/SystemErrorWang/White-box-Cartoonization)

[2] Python implementation of the pencil drawing by sketch and tone algorithm. [Github Repo Link](https://github.com/taldatech/image2pencil-drawing)



