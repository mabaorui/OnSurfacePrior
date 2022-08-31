# Reconstructing Surfaces for Sparse Point Clouds with On-Surface Priors (CVPR 2022)

<h2 align="center"><a href="https://mabaorui.github.io/">Personal Web Pages</a> | <a href="https://arxiv.org/abs/2204.10603">Paper</a> | <a href="https://mabaorui.github.io/-OnSurfacePrior_project_page/">Project Page</a></h2>

This repository contains the code to reproduce the results from the paper.
[Reconstructing Surfaces for Sparse Point Clouds with On-Surface Priors](https://arxiv.org/abs/2204.10603).

You can find detailed usage instructions for training your own models and using pretrained models below.

If you find our code or paper useful, please consider citing

    @inproceedings{On-SurfacePriors,
        title = {Reconstructing Surfaces for Sparse Point Clouds with On-Surface Priors},
        author = {Baorui, Ma and Yu-Shen, Liu and Zhizhong, Han},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2022}
    }

## Pytorch Version
This work was originally implemented by tensorflow, pytroch version of the code will be released soon that is easier to use.

Related work
```bash
Pytorch 
https://github.com/mabaorui/NeuralPull-Pytorch
Tensorflow
https://github.com/mabaorui/NeuralPull
https://github.com/mabaorui/OnSurfacePrior
https://github.com/mabaorui/PredictableContextPrior
```


## Surface Reconstruction Demo
<p align="left">
  <img src="img/ParisStreet_part.jpg" width="780" />
</p>

<p align="left">
  <img src="img/plane.png" width="780" />
</p>

<p align="left">
  <img src="img/scene.png" width="780" />
</p>

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `tf` using
```
conda env create -f tf.yaml
conda activate tf
```
## Training
You can train a new network from pre-train On-Surface Prior Networks, run

```
python onSurPrior.py --data_dir ./data/ --out_dir ./train_net/ --CUDA 0 --INPUT_NUM 500 --epoch 30000 --input_ply_file input.ply --train
```
You should put the point cloud file(--input_ply_file, only ply format) into the '--out_dir' folder, '--INPUT_NUM' is the number of points in the '--input_ply_file'.
## Test
You can extract the mesh model from the trained network, run
```
python onSurPrior.py --data_dir ./data/ --out_dir ./train_net/ --CUDA 0 --INPUT_NUM 500 --epoch 30000 --input_ply_file input.ply --test
```

## ToDo
In different datasets or your own data, because of the variation in point cloud density, this ['0.25' parameter](https://github.com/mabaorui/OnSurfacePrior/blob/d53bf3a7bc88837e2974ddc1fd0700ecc2641ade/onSurPrior.py#L425) has a very strong influence on the final result, which controls the distance between the query points and the point cloud. So if you want to get better results, you should adjust this parameter. We give '0.25' here as a reference value, and this value can be used for most object-level reconstructions. For the scene dataset, we will later publish the reference values for the hyperparameter settings for the scene dataset.
