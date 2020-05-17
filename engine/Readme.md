This folder contains main program.

Compiling
---
To compile engine you need linux system (it uses only posix threads and getopt, porting to other system shouldn't be hard) with CUDA libraries and Nvidia GPU. You may need to adjust Makefile $CUDAPATH.

Usage
---
```
USAGE: ./checkers-cpu [OPTIONS] - find best move in checkers
Options:
-t <time> - set computing timeout (in miliseconds). Default is 1000ms=1s
-p <player> - which player has move. 0 - light, 1 - dark. Default is 1.
After run program read from stdin board state where:
"." - empty square
"l" - light men
"L" - light knight
"d" - dark men
"D" - dark knight
Example (starting board):
.d.d.d.d
d.d.d.d.
.d.d.d.d
........
........
l.l.l.l.
.l.l.l.l
l.l.l.l.
(rows are in stdin order, first row is on the top)
```