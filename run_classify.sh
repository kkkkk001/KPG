python -u KPGCN_train.py --datasetname ${1} --num_class ${2} --res_log_file GCN_log/${1}.txt --model_save_path GCN/${1} --main_data_path rl_data/${1} --modelname GCN --best_epoch_path best_epoch/all.json  --return_prob 0 
python -u bert_main.py --datasetname ${1} --num_class ${2} --res_log_file bert_log/${1}.txt --text full --early_stopping 1 --model_save_path bert/${1}
python -u combine_inference.py --datasetname ${1} --num_class ${2} --rl_data_main_path rl_data/${1} --bert_model_path bert/${1} --GCN_model GCN --GCN_model_path GCN/${1} --temp 4
