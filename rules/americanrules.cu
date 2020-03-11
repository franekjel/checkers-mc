#include "americanrules.h"

__device__ __host__ void AmericanRules::getMovesLightPos(char board[], Move moves[], int& captures, int8_t x, int8_t y, int& n)
{
    if (board[y * boardSize + x] == 'l' || board[y * boardSize + x] == 'L')
    {
        //men move
        if (captures == 0)
        { //when there is capture possibility we can't do normal move
            if (x > 0 && y > 0 && board[(y - 1) * boardSize + x - 1] == '.')
            {
                moves[n][0] = x;
                moves[n][1] = y;
                moves[n][2] = x - 1;
                moves[n][3] = y - 1;
                n++;
            }
            if (x < boardSize - 1 && y > 0 && board[(y - 1) * boardSize + x + 1] == '.')
            {
                moves[n][0] = x;
                moves[n][1] = y;
                moves[n][2] = x + 1;
                moves[n][3] = y - 1;
                n++;
            }
        }
        //men capture
        if (y > 1)
        {
            if (x > 1 && board[(y - 2) * boardSize + x - 2] == '.' && (board[(y - 1) * boardSize + x - 1] == 'd' || board[(y - 1) * boardSize + x - 1] == 'D'))
            {
                moves[n][0] = x;
                moves[n][1] = y;
                moves[n][2] = x - 2;
                moves[n][3] = y - 2;
                n++;
                captures++;
            }
            if (x < boardSize - 2 && board[(y - 2) * boardSize + x + 2] == '.' && (board[(y - 1) * boardSize + x + 1] == 'd' || board[(y - 1) * boardSize + x + 1] == 'D'))
            {
                moves[n][0] = x;
                moves[n][1] = y;
                moves[n][2] = x + 2;
                moves[n][3] = y - 2;
                n++;
                captures++;
            }
        }

        if (board[y * boardSize + x] == 'L')
        { //king
            if (captures == 0)
            { //normal king move
                if (x > 0 && y < 7 && board[(y + 1) * boardSize + x - 1] == '.')
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x - 1;
                    moves[n][3] = y + 1;
                    n++;
                }
                if (x < 7 && y < 7 && board[(y + 1) * boardSize + x + 1] == '.')
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x + 1;
                    moves[n][3] = y + 1;
                    n++;
                }
            }
            //king capture
            if (y < boardSize - 2)
            {
                if (x > 1 && board[(y + 2) * boardSize + x - 2] == '.' && (board[(y + 1) * boardSize + x - 1] == 'd' || board[(y + 1) * boardSize + x - 1] == 'D'))
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x - 2;
                    moves[n][3] = y + 2;
                    n++;
                    captures++;
                }
                if (x < boardSize - 2 && board[(y + 2) * boardSize + x + 2] == '.' && (board[(y + 1) * boardSize + x + 1] == 'd' || board[(y + 1) * boardSize + x + 1] == 'D'))
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x + 2;
                    moves[n][3] = y + 2;
                    n++;
                    captures++;
                }
            }
        } //king
    }
}

__device__ __host__ void AmericanRules::getMovesDarkPos(char board[], Move moves[], int& captures, int8_t x, int8_t y, int& n)
{
    if (board[y * boardSize + x] == 'd' || board[y * boardSize + x] == 'D')
    {
        if (captures == 0)
        { //when there is capture possibility we can't do normal move
            if (x > 0 && y < 7 && board[(y + 1) * boardSize + x - 1] == '.')
            {
                moves[n][0] = x;
                moves[n][1] = y;
                moves[n][2] = x - 1;
                moves[n][3] = y + 1;
                n++;
            }
            if (x < 7 && y < 7 && board[(y + 1) * boardSize + x + 1] == '.')
            {
                moves[n][0] = x;
                moves[n][1] = y;
                moves[n][2] = x + 1;
                moves[n][3] = y + 1;
                n++;
            }
        }
        if (y < boardSize - 2)
        { //men capture
            if (x > 1 && board[(y + 2) * boardSize + x - 2] == '.' && (board[(y + 1) * boardSize + x - 1] == 'l' || board[(y + 1) * boardSize + x - 1] == 'L'))
            {
                moves[n][0] = x;
                moves[n][1] = y;
                moves[n][2] = x - 2;
                moves[n][3] = y + 2;
                n++;
                captures++;
            }
            if (x < boardSize - 2 && board[(y + 2) * boardSize + x + 2] == '.' && (board[(y + 1) * boardSize + x + 1] == 'l' || board[(y + 1) * boardSize + x + 1] == 'L'))
            {
                moves[n][0] = x;
                moves[n][1] = y;
                moves[n][2] = x + 2;
                moves[n][3] = y + 2;
                n++;
                captures++;
            }
        }

        if (board[y * boardSize + x] == 'D')
        { //king
            if (captures == 0)
            { //normal king move
                if (x > 0 && y > 0 && board[(y - 1) * boardSize + x - 1] == '.')
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x - 1;
                    moves[n][3] = y - 1;
                    n++;
                }
                if (x < 7 && y > 0 && board[(y - 1) * boardSize + x + 1] == '.')
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x + 1;
                    moves[n][3] = y - 1;
                    n++;
                }
            }
            if (y > 1)
            { //king capture
                if (x > 1 && board[(y - 2) * boardSize + x - 2] == '.' && (board[(y - 1) * boardSize + x - 1] == 'l' || board[(y - 1) * boardSize + x - 1] == 'L'))
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x - 2;
                    moves[n][3] = y - 2;
                    n++;
                    captures++;
                }
                if (x < boardSize - 2 && board[(y - 2) * boardSize + x + 2] == '.' && (board[(y - 1) * boardSize + x + 1] == 'l' || board[(y - 1) * boardSize + x + 1] == 'L'))
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x + 2;
                    moves[n][3] = y - 2;
                    n++;
                    captures++;
                }
            }
        } //king
    }
}

__device__ __host__ int AmericanRules::checkResult(char board[])
{
    int l = 0, d = 0;
    for (int x = 0; x < boardSize; x++)
    {
        for (int y = 0; y < boardSize; y++)
        {
            if (board[y * boardSize + x] == 'd' || board[y * boardSize + x] == 'D')
                d++;
            if (board[y * boardSize + x] == 'l' || board[y * boardSize + x] == 'L')
                l++;
        }
    }
    if (l == 0)
        return 1;
    if (d == 0)
        return 0;
    return -1;
}
