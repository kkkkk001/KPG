## KPG: Rumor Detection on Social Media with Reinforcement Learning-based Key Propagation Graph Generator

### Running pipeline 

1. Running KPG to generate the key propagation graphs, which will be stored in folder *rl_data*.
```
bash run.sh Twitter16 4
```

2. Using the key propagation graphs to obtain the final results. 
```
bash run_classify.sh Twitter16 4
```


### Datasets
For datasets Twitter15, Twitter16 and Pheme, we have already construct the propagation graphs and explicitly stored in folder *data*. 

For our newly collected Weibo22 dataset introduced in this paper, we provide text files and processing scripts instead of the complete graph data due to its large size. To test on this dataset, please first unzip two .zip files under the *data/Weibo* directory (unzip into the same directory), then run the following command to extract propagation graphs from the text files.

```
python Process/getWeibograph.py
```

If you find our dataset helpful, please cite our paper: 
```
@article{zhang2024kpg,
  title={KPG: Key Propagation Graph Generator for Rumor Detection based on Reinforcement Learning},
  author={Zhang, Yusong and Xie, Kun and Zhang, Xingyi and Dong, Xiangyu and Wang, Sibo},
  journal={arXiv preprint arXiv:2405.13094},
  year={2024}
}
```