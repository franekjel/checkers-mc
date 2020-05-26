Checkers-mc
----
Program to find best move in checkers using Monte-Carlo Tree Search and CUDA. 

Usage
---
You can use program directly (check engine folder) or as CheckerBoard engine.
 - CheckerBoard - this folder contains eygilbert/CheckerBoard.
 - cmcserver - small program in go to host main engine for CheckerBoard
 - engine - main program, MCTS checkers engine
 - mcts-client - .dll to be used by CheckerBoard
With this you can compile cmcserver and engine on one computer with good graphics card (and linux) then on another computer with windows run CheckerBoard with mcts-client as engine

Program strength
---
Program is better than unprofessional player (like me) or some random online checkers programs, but loses to more powerfull engines like KingsRow or Cake. It seems that MCTS is not the best algorithm for checkers
