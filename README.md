## KPG

There two phases are needed to run KPG.

1. Running KPG to generate the key propagation graphs, which are stroed in folder *rl_data*
```
bash run.sh Twitter16 4
```

2. Using the key propagation graphs to obtain the final resultes. 
```
bash run_classify.sh Twitter16 4
```

For Weibo22 dataset, please first unzip files in data/Weibo, and the run the following commands to generate the propagation graphs from the text files.

```
python Process/getWeibograph.py
```