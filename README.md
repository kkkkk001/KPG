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

Note: For the Weibo22 dataset, please first unzip files in *data/Weibo*, and then run the following command to extract the propagation graphs from the text files.

```
python Process/getWeibograph.py
```
