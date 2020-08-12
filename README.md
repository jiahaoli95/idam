# Code for ECCV 2020 paper "Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration" (IDAM)
This is the official code repository for ECCV 2020 paper "Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration" [[arxiv](https://arxiv.org/abs/1910.10328)]. The code is largely adapted from https://github.com/WangYueFt/dcp and https://github.com/WangYueFt/prnet.

[![model.png](https://s7.gifyu.com/images/model.png)](https://gifyu.com/image/WkbF)

[![elimination.png](https://s7.gifyu.com/images/elimination.png)](https://gifyu.com/image/WkbY)

## Environment
We recommend using Anaconda to set up common packages such as numpy and scipy. The following packages are required but not installed by default by Anaconda. We list the version numbers of the packages that we use on our machine, but other versions should work, possibly with minor modification:

Pytorch 1.4.0

Open3D 0.7.0.0

tqdm

## Usage
The easiest way to run the code is using the following command
```
python main.py --exp_name exp
```
This command will run an experiment on the ModelNet40 dataset (automatically downloaded) with all the options set to default. You can see at the end of main.py a list of options that can be used to control hyperparameters of the model and experiment settings. The comments in the file should be enough to understand them.

## Citation
If you want to use it in your work, please cite it as

	@InProceedings{idam,
	  title={Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration},
	  author={Li, Jiahao and Zhang, Changhao and Xu, Ziyao and Zhou, Hangning and Zhang, Chi},
	  booktitle = {European Conference on Computer Vision (ECCV)},
	  year={2020}
	}
