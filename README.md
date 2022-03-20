## Reconstructing Surfaces for Sparse Point Clouds with On-Surface Priors(CVPR 2022)

This repository contains the code to reproduce the results from the paper.
[Neural-Pull: Learning Signed Distance Functions from Point Clouds by Learning to Pull Space onto Surfaces](https://arxiv.org/abs/2011.13495).

You can find detailed usage instructions for training your own models and using pretrained models below.

If you find our code or paper useful, please consider citing

    @inproceedings{NeuralPull,
        title = {Reconstructing Surfaces for Sparse Point Clouds with On-Surface Priors},
        author = {Baorui, Ma and Yu-shen, Liu and Zhizhong, Han},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2022}
    }

## Surface Reconstruction Demo
<p align="left">
  <img src="img/dragon.gif" width="780" />
</p>

<p align="left">
  <img src="img/plane_sur.gif" width="780" />
</p>

## Single Image Reconstruction Demo
<p align="center">
  <img src="img/002.jpg" width="380" />
</p>

<p align="left">
  <img src="img/plane_svg.gif" width="780" />
</p>

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `tensorflow1` using
```
conda env create -f NeuralPull.yaml
conda activate tensorflow1
```

Next, for evaluation of the models,compile the extension modules, which are provided by [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks).
You can do this via
```
python setup.py build_ext --inplace
```

To compile the dmc extension, you have to have a cuda enabled device set up.
If you experience any errors, you can simply comment out the `dmc_*` dependencies in `setup.py`.
You should then also comment out the `dmc` imports in `im2mesh/config.py`.


## Dataset and pretrained model

1. You can [download](https://drive.google.com/drive/folders/1qre9mgJNCKiX11HnZO10qMZMmPv_gnh3?usp=sharing) our preprocessed data and pretrained model.Included in the link:


   --Our pre-train model on ABC and FAMOUS dataset.
   
   
   --Preprocessing data of ABC and FAMOUS(sample points and ground truth points).
   
   
   --Our reconstruction results.
   
   
2. To make it easier for you to test the code, we have prepared exmaple data in the exmaple_data folder.

### Building the dataset
Alternatively, you can also preprocess the dataset yourself.
To this end, you have to follow the following steps:
* Put your own pointcloud files in 'input_dir' folder, each pointcloud file in a separate .xyz.npy file. 
* Set an empty folder 'out_dir' to place the processed data, note, the folder need to be empty, because this folder will be deleted before the program runs.

You are now ready to build the dataset:
```
python sample_query_point --out_dir /data1/mabaorui/AtlasNetOwn/data/plane_precompute_2/ --CUDA 0 --dataset other --input_dir ./data/abc_noisefree/04_pts/ 
``` 

## Training
You can train a new network from scratch, run
1. Surface Reconstruction

```
python NeuralPull.py --data_dir /data1/mabaorui/AtlasNetOwn/data/plane_precompute_2/ --out_dir /data1/mabaorui/AtlasNetOwn/plane_cd_sur/ --class_idx 02691156 --train --dataset shapenet
```
2. Single Image Reconstruction
```
python NeuralPull_SVG.py --data_dir /data1/mabaorui/AtlasNetOwn/data/plane_precompute_2/ --out_dir /data1/mabaorui/AtlasNetOwn/plane_cd_sur/ --class_idx 02691156 --train --class_name plane
```
3. Train the dataset yourself
```
python NeuralPull.py --data_dir /data1/mabaorui/AtlasNetOwn/data/plane_precompute_2/ --out_dir /data1/mabaorui/AtlasNetOwn/plane_cd_sur/ --class_idx 02691156 --train --dataset other
```


## Evaluation
For evaluation of the models and generation meshes using a trained model, use
1. Surface Reconstruction

```
python NeuralPull.py --data_dir /data1/mabaorui/AtlasNetOwn/data/plane_precompute_2/ --out_dir /data1/mabaorui/AtlasNetOwn/plane_cd_sur/ --class_idx 02691156 --dataset shapenet
```

2. Single Image Reconstruction

```
python NeuralPull_SVG.py --data_dir /data1/mabaorui/AtlasNetOwn/data/plane_precompute_2/ --out_dir /data1/mabaorui/AtlasNetOwn/plane_cd_sur/ --class_idx 02691156 --class_name plane
```
3. Evaluation the dataset yourself
```
python NeuralPull.py --data_dir /data1/mabaorui/AtlasNetOwn/data/plane_precompute_2/ --out_dir /data1/mabaorui/AtlasNetOwn/plane_cd_sur/ --class_idx 02691156 --dataset other
```

## Script Parameters Explanation

|Parameters  |Description |
|:---------------|:----------------------------|
|train            |train or test a network.      |
|data_dir           |preprocessed data.      |
|out_dir           |store network parameters when training or to load pretrained network parameters when testing.      |
|class_idx          |the class to train or test when using shapenet dataset, other dataset, default.      |
|class_name           |the class to train or test when using shapenet dataset, other dataset, default.       |
|dataset           |shapenet,famous,ABC or other(your dataset)      |

## Pytorch Implementation of Neural-Pull

Notably, [the code in Pytorch](https://github.com/wzxshhz123/NeuralPull_pytorch) implementation is not released by the official lab, it is achieved by @wzxshgz123's diligent work. His intention is only to provide references to researchers who are interested in Pytorch implementation of Neural-Pull. There is no doubt that his unconditional dedication should be appreciated. 
