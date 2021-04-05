#!/bin/bash
(echo $PWD && cd .. && nohup python -m src.actors.tester $1 $2 > ./scripts/log_tester.log) &
(echo $PWD && cd .. && nohup python -m src.actors.aggregator $1 $2 > ./scripts/log_agg.log) &
