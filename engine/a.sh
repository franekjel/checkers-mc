#!/bin/bash

for i in {1..10}
do
	./checkers -t100 -p1 < start.txt
done
