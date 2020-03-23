#include <cstdio>
#include <ctime>
extern "C"
{
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
}

#include "kernel.h"
#include "rules/americanrules.h"
#include "rules/rules.h"

void handler(int sig)
{
    void* array[10];
    size_t size;
    size = backtrace(array, 10);
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

void usage(char* pname)
{
    fprintf(stderr, "USAGE: %s [OPTIONS] - find best move in checkers \n\
Options:\n\
-t <time> - set computing timeout (in miliseconds). Default is 1000ms=1s\n\
-p <player> - which player has move. 0 - light, 1 - dark. Default is 1.\n\
After run program read from stdin board state in format:\n\
xxxxxxxx\n\
xxxxxxxx\n\
xxxxxxxx\n\
xxxxxxxx\n\
xxxxxxxx\n\
xxxxxxxx\n\
xxxxxxxx\n\
xxxxxxxx\n\
where \"x\" may be:\n\
\".\" - empty square\n\
\"l\" - light men\n\
\"L\" - light knight\n\
\"d\" - dark men\n\
\"D\" - dark knight\n\
Example (starting board):\n\
.d.d.d.d\n\
d.d.d.d.\n\
.d.d.d.d\n\
........\n\
........\n\
l.l.l.l.\n\
.l.l.l.l\n\
l.l.l.l.\n\
",
        pname);
    exit(EXIT_FAILURE);
}

//very basic checking of board correctness
void checkBoard(char* board, int n, char* argv[])
{
    for (int x = 0; x < n; x++)
    {
        for (int y = 0; y < n; y++)
        {
            if ((x + y) % 2 == 0 && board[8 * y + x] != '.')
            {
                printf("Piece on white square!\n");
                usage(argv[0]);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    signal(SIGSEGV, handler);
    int timeout = 1000;
    char c;
    int player = 0;
    while ((c = getopt(argc, argv, "t:p:g")) != -1)
    {
        switch (c)
        {
        case 't':
            sscanf(optarg, "%d", &timeout);
            break;
        case 'p':
            sscanf(optarg, "%d", &player);
            break;
        default:
            usage(argv[0]);
        }
    }
    char board[8][8];
    int i = 0;
    while (i < 64)
    {
        board[i / 8][i % 8] = getchar();
        if (board[i / 8][i % 8] == '.' || board[i / 8][i % 8] == 'd' || board[i / 8][i % 8] == 'D' || board[i / 8][i % 8] == 'l' || board[i / 8][i % 8] == 'L')
            i++;
    }
    //checkBoard(board, n, argv);

    findMoveGPU<AmericanRules, 8, 48>(board, timeout, player);

    return 0;
}
