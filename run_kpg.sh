
python train_gcn.py ${1} ${2} pretrainedGCN 
python -u KPG_main_final.py $1 0 $2  res_log/${1}_0_${2}_.txt
python -u KPG_main_final.py $1 1 $2  res_log/${1}_1_${2}_.txt
python -u KPG_main_final.py $1 2 $2  res_log/${1}_2_${2}_.txt
python -u KPG_main_final.py $1 3 $2  res_log/${1}_3_${2}_.txt
python -u KPG_main_final.py $1 4 $2  res_log/${1}_4_${2}_.txt
