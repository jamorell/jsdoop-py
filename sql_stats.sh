#!/bin/bash
nohup python -m src.utils.load_stats_to_sql $1 $2 &
nohup python -m src.utils.load_acc_to_sql $1 $2 &

