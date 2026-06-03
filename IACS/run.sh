nohup python -u main.py --data_set twitter --task_size 9 --num_shots 4 --train_task_num 128 --valid_task_num 32 --test_task_num 32 > result2.out 2>&1 &
#nohup python -u main_QDGNN_attr.py > result.out 2>&1 &
#train_task_num、valid_task_num、test_task_num没有用，主要在代码中的用处是指定任务数