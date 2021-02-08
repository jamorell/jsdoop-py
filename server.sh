#!/bin/bash
nohup python -m src.actors.tester $1 $2 > log_tester.log &
nohup python -m src.actors.aggregator $1 $2 > log_agg.log &
nohup python -m src.utils.load_stats_to_sql $1 $2 > log_stats.log &
nohup python -m src.utils.load_acc_to_sql $1 $2 > log_acc.log &

