#pragma once
#include "rules.h"

//implemenation of american rules. Check rules.h for description
class AmericanRules
{
public:
    static __device__ __host__ void getMovesLightPos(char board[8][8], Move moves[48], int& captures, int8_t x, int8_t y, int& n)
    {
        if (board[x][y] == 'l' || board[x][y] == 'L')
        {
            //men move
            if (captures == 0)
            { //when there is capture possibility we can't do normal move
                if (x > 0 && y > 0 && board[x - 1][y - 1] == '.')
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x - 1;
                    moves[n][3] = y - 1;
                    n++;
                }
                if (x < 7 && y > 0 && board[x + 1][y - 1] == '.')
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
                if (x > 1 && board[x - 2][y - 2] == '.' && (board[x - 1][y - 1] == 'd' || board[x - 1][y - 1] == 'D'))
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x - 2;
                    moves[n][3] = y - 2;
                    n++;
                    (captures)++;
                }
                if (x < 6 && board[x + 2][y - 2] == '.' && (board[x + 1][y - 1] == 'd' || board[x + 1][y - 1] == 'D'))
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x + 2;
                    moves[n][3] = y - 2;
                    n++;
                    (captures)++;
                }
            }

            if (board[x][y] == 'L')
            { //king
                if (captures == 0)
                { //normal king move
                    if (x > 0 && y < 7 && board[x - 1][y + 1] == '.')
                    {
                        moves[n][0] = x;
                        moves[n][1] = y;
                        moves[n][2] = x - 1;
                        moves[n][3] = y + 1;
                        n++;
                    }
                    if (x < 7 && y < 7 && board[x + 1][y + 1] == '.')
                    {
                        moves[n][0] = x;
                        moves[n][1] = y;
                        moves[n][2] = x + 1;
                        moves[n][3] = y + 1;
                        n++;
                    }
                }
                //king capture
                if (y < 6)
                {
                    if (x > 1 && board[x - 2][y + 2] == '.' && (board[x - 1][y + 1] == 'd' || board[x - 1][y + 1] == 'D'))
                    {
                        moves[n][0] = x;
                        moves[n][1] = y;
                        moves[n][2] = x - 2;
                        moves[n][3] = y + 2;
                        n++;
                        (captures)++;
                    }
                    if (x < 6 && board[x + 2][y + 2] == '.' && (board[x + 1][y + 1] == 'd' || board[x + 1][y + 1] == 'D'))
                    {
                        moves[n][0] = x;
                        moves[n][1] = y;
                        moves[n][2] = x + 2;
                        moves[n][3] = y + 2;
                        n++;
                        (captures)++;
                    }
                }
            } //king
        }
    }

    static __device__ __host__ void getMovesDarkPos(char board[8][8], Move moves[48], int& captures, int8_t x, int8_t y, int& n)
    {
        if (board[x][y] == 'd' || board[x][y] == 'D')
        {
            if (captures == 0)
            { //when there is capture possibility we can't do normal move
                if (x > 0 && y < 7 && board[x - 1][y + 1] == '.')
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x - 1;
                    moves[n][3] = y + 1;
                    n++;
                }
                if (x < 7 && y < 7 && board[x + 1][y + 1] == '.')
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x + 1;
                    moves[n][3] = y + 1;
                    n++;
                }
            }
            if (y < 6)
            { //men capture
                if (x > 1 && board[x - 2][y + 2] == '.' && (board[x - 1][y + 1] == 'l' || board[x - 1][y + 1] == 'L'))
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x - 2;
                    moves[n][3] = y + 2;
                    n++;
                    (captures)++;
                }
                if (x < 6 && board[x + 2][y + 2] == '.' && (board[x + 1][y + 1] == 'l' || board[x + 1][y + 1] == 'L'))
                {
                    moves[n][0] = x;
                    moves[n][1] = y;
                    moves[n][2] = x + 2;
                    moves[n][3] = y + 2;
                    n++;
                    (captures)++;
                }
            }

            if (board[x][y] == 'D')
            { //king
                if (captures == 0)
                { //normal king move
                    if (x > 0 && y > 0 && board[x - 1][y - 1] == '.')
                    {
                        moves[n][0] = x;
                        moves[n][1] = y;
                        moves[n][2] = x - 1;
                        moves[n][3] = y - 1;
                        n++;
                    }
                    if (x < 7 && y > 0 && board[x + 1][y - 1] == '.')
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
                    if (x > 1 && board[x - 2][y - 2] == '.' && (board[x - 1][y - 1] == 'l' || board[x - 1][y - 1] == 'L'))
                    {
                        moves[n][0] = x;
                        moves[n][1] = y;
                        moves[n][2] = x - 2;
                        moves[n][3] = y - 2;
                        n++;
                        (captures)++;
                    }
                    if (x < 6 && board[x + 2][y - 2] == '.' && (board[x + 1][y - 1] == 'l' || board[x + 1][y - 1] == 'L'))
                    {
                        moves[n][0] = x;
                        moves[n][1] = y;
                        moves[n][2] = x + 2;
                        moves[n][3] = y - 2;
                        n++;
                        (captures)++;
                    }
                }
            } //king
        }
    }
    static __device__ __host__ int checkResult(char board[8][8])
    {
        int l = 0, d = 0;
        for (int x = 0; x < 8; x++)
        {
            for (int y = 0; y < 8; y++)
            {
                if (board[x][y] == 'd' || board[x][y] == 'D')
                    d++;
                if (board[x][y] == 'l' || board[x][y] == 'L')
                    l++;
            }
        }
        if (l == 0)
            return 1;
        if (d == 0)
            return 0;
        return -1;
    }
};
