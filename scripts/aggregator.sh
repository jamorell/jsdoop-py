#!/bin/bash
echo $1
echo $2
cd .. && python -m src.actors.aggregator $1 $2

