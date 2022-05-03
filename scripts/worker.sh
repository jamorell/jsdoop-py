#!/bin/bash
echo $1
echo $2
echo $3
echo $4
echo "ok"
cd .. && python -m src.actors.worker $1 $2 $3 $4
