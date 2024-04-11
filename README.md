## KPG

Two phases are needed to run KPG.

1. Running KPG to generate the key propagation graphs, which are stored in folder *rl_data*
```
bash run.sh Twitter16 4
```

2. Using the key propagation graphs to obtain the final results. 
```
bash run_classify.sh Twitter16 4
```

Note: For the Weibo22 dataset, please first unzip files in *data/Weibo*, and then run the following commands to generate the propagation graphs from the text files.

```
python Process/getWeibograph.py
```
