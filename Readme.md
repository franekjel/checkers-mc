Checkers-mc
----
Program to find best move in checkers using Monte-Carlo Tree Search and CUDA. 

Usage
---
You can use engine directly (check engine folder) or let CheckerBoard use it. Therefore there are folders:
 - CheckerBoard - this folder contains eygilbert/CheckerBoard. It can be use to compare strenght and performance between engines
 - cmcserver - small program in go to host main engine for CheckerBoard
 - engine - main program, MCTS checkers engine
 - mcts-client - .dll for CheckerBoard.
With this you can compile cmcserver and engine on one computer with good graphics card (and eg. linux) then on another computer with windows run CheckerBoard (CheckerBoard is windows program) with mcts-client as engine
