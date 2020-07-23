# Code for Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration (ECCV 2020)
This is the official code repository for ECCV 2020 paper "Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration" [[arxiv](https://arxiv.org/abs/1910.10328)]. The code is largely adapted from https://github.com/WangYueFt/dcp and https://github.com/WangYueFt/prnet.

## Environment
We recommend using Anaconda to set up common packages such as numpy and scipy. The following packages are required but not installed by default by Anaconda. We list the version numbers of the packages that we use on our machine, but other versions should work, possibly with minor modification:

Pytorch 1.4.0

Open3D 0.7.0.0

tqdm

## Use
The easiest way to run the code is using the following command
```
python main.py --exp_name exp
```
This command will run an experiment with all the options set to default. You can see at the end of main.py a list of options that can be used to control hyperparameters of the model and experiment settings. The comments in the file should be enough to understand them.
