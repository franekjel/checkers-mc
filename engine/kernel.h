#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <string.h>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

extern "C"
{
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
}

#include "rules/americanrules.h"
#include "rules/rules.h"

// for Testla T4 (2560 CUDA Cores)
#define THREADS 10
#define BLOCK 256

int Player = 0;
int device;

std::atomic<int> kernels;

volatile sig_atomic_t running = 1;

//One move - from.x, from.y, to.x, to.y It have operator=
typedef int8_t Move[4];

struct TreeNode
{
    TreeNode* parent;
    int position;
    std::atomic<int> wins;
    std::atomic<int> games;
    TreeNode** children;
    Move* moves;
    int movesN;
    TreeNode(TreeNode* p)
    {
        parent = p;
        position = -1;
        wins = 0;
        games = 0;
        children = nullptr;
        movesN = -1;
    }
};

template <int TBoardSize>
struct ThreadData
{
    char board[TBoardSize][TBoardSize];
    int player;
    char* boards;
    int* positions;
    float* results;
    curandState* states;
    pthread_mutex_t* rootMutex;
    TreeNode* root;
    cudaStream_t stream;
};

static void FreeMemory(TreeNode* root)
{
    for (int i = 0; i < root->movesN; i++)
        FreeMemory(root->children[i]);
    delete root;
}

static int chooseNode(TreeNode* cur, TreeNode* root)
{
    float max = 0;
    int maxi = 0;
    for (int i = 0; i < cur->movesN; i++)
    {
        if (cur->children[i]->games == 0)
            return i;
        else
        {
            float u = float(cur->children[i]->wins) / float(cur->children[i]->games); // w/g
            u += 7 * sqrtf(logf(root->games) / float(cur->children[i]->games));
            u += rand() % 2;
            if (u > max)
            {
                max = u;
                maxi = i;
            }
        }
    }
    return maxi;
}

static int getBestMove(TreeNode* cur)
{
    int max = 0;
    int maxi = -1;
    for (int i = 0; i < cur->movesN; i++)
    {
        float g = cur->children[i]->games.load();
        float w = cur->children[i]->wins.load();
        if (w / g > max)
        {
            max = w / g;
            maxi = i;
        }
    }
    return maxi;
}

template <class TRules, int TBoardSize, int TMaxMoves>
static void Expand(TreeNode* node, char board[TBoardSize][TBoardSize], int player)
{
    Move moves[TMaxMoves];
    int captures = 0, n = 0;
    if (node->position != -1)
    {
        if (player)
            TRules::getMovesDarkPos(board, moves, captures, (node->position) % TBoardSize, (node->position) / TBoardSize, n);
        else
            TRules::getMovesLightPos(board, moves, captures, (node->position) % TBoardSize, (node->position) / TBoardSize, n);
    } else
    {
        if (player) //0-light 1-dark
            n = genMovesDark<TRules, TBoardSize, TMaxMoves>(board, moves, captures);
        else
            n = genMovesLight<TRules, TBoardSize, TMaxMoves>(board, moves, captures);
    }
    if (captures > 0)
        n = filterMoves<TMaxMoves>(moves, n);
    node->movesN = n;
    node->children = new TreeNode*[n];
    node->moves = new Move[n];
    for (int i = 0; i < n; i++)
    {
        memcpy(node->moves[i], moves[i], sizeof(Move));
        node->children[i] = new TreeNode(node);
        if (captures > 0)
        { //if there are capturing moves all are capturing. So we check if after capture we have another possibility to capture
            char b[TBoardSize][TBoardSize];
            memcpy(b, board, TBoardSize * TBoardSize * sizeof(char));
            b[node->moves[i][2]][node->moves[i][3]] = b[node->moves[i][0]][node->moves[i][1]];
            b[node->moves[i][0]][node->moves[i][1]] = '.';
            b[(node->moves[i][0] + node->moves[i][2]) / 2][(node->moves[i][1] + node->moves[i][3]) / 2] = '.';
            int m = 0;
            Move mvs[TMaxMoves];
            int cptr = 0;
            if (player)
                TRules::getMovesDarkPos(b, mvs, cptr, node->moves[i][2], node->moves[i][3], m);
            else
                TRules::getMovesLightPos(b, mvs, cptr, node->moves[i][2], node->moves[i][3], m);
            if (cptr > 0)
            {
                node->children[i]->position = node->moves[i][2] + TBoardSize * node->moves[i][3];
            }
        }
    }
}

//simulation starts in node with children, we choose one and do random game. Then backpropagate result
//__global__ void __launch_bounds__(BLOCK) MCTSSimulation(char* boards, int* positions, curandState* states, int player, int orginalPlayer, float* results)
template <class TRules, int TBoardSize, int TMaxMoves>
__global__ void MCTSSimulation(char* boards, int* positions, curandState* states, int player, int orginalPlayer, float* results)
{
    const int boardArea = TBoardSize * TBoardSize;
    int idx = threadIdx.x;
    curandState state = states[idx];
    char board[TBoardSize][TBoardSize];
    memcpy(board, boards + (boardArea * idx), boardArea * sizeof(char));
    int dep = 0; //depth
    float r = 0; //result
    Move moves[TMaxMoves];
    int captures = 0;
    int xpos = positions[idx] % TBoardSize, ypos = positions[idx] / TBoardSize;
    positions[idx] = -1;
    while (dep < 120)
    { //NOTE: magic number! - max depth of simulation
        int n = 0;
        captures = 0;
        if (xpos > -1)
        {
            if (player)
                TRules::getMovesDarkPos(board, moves, captures, xpos, ypos, n);
            else
                TRules::getMovesLightPos(board, moves, captures, xpos, ypos, n);
        } else
        {
            if (player)
                n = genMovesDark<TRules, TBoardSize, TMaxMoves>(board, moves, captures);
            else
                n = genMovesLight<TRules, TBoardSize, TMaxMoves>(board, moves, captures);
        }
        if (n == 0)
        {
            r = player ^ orginalPlayer;
            r *= 2;
            break;
        }

        if (captures > 0)
            n = filterMoves<TMaxMoves>(moves, n);
        int i = curand(&state) % n;
        if (dep == 0)
            positions[idx] = i;
        board[moves[i][2]][moves[i][3]] = board[moves[i][0]][moves[i][1]];
        board[moves[i][0]][moves[i][1]] = '.';
        if (moves[i][0] - moves[i][2] == 2 || moves[i][0] - moves[i][2] == -2)
        { //capturign move
            board[(moves[i][2] + moves[i][0]) / 2][(moves[i][3] + moves[i][1]) / 2] = '.';
            xpos = moves[i][2];
            ypos = moves[i][3];
            int cptr = 0, m = 0;
            if (player)
                TRules::getMovesDarkPos(board, moves, cptr, xpos, ypos, m);
            else
                TRules::getMovesLightPos(board, moves, cptr, xpos, ypos, m);
            m = filterMoves<TMaxMoves>(moves, m);
            if (m == 0)
            {
                xpos = -1;
                player = player ^ 1;
            }
        } else
            player = player ^ 1;
        int c = TRules::checkResult(board);
        if (c != -1)
        {
            if (c == orginalPlayer)
                r = 2;
            break;
        }
        dep++;
    } //simulation
    if (dep == 120)
        r = 1;

    results[idx] = r;
}

template <class TRules, int TBoardSize, int TMaxMoves>
static TreeNode* MCTSSelectionAndExpansion(char board[TBoardSize][TBoardSize], TreeNode* root, int* player)
{
    int orginalPlayer = *player;
    int N0 = 20; //NOTE: magic number! - how fast node expands
    TreeNode* cur = root;
    while (cur->movesN != -1 && cur->games.load() > N0)
    {
        int i = chooseNode(cur, root);
        cur->games.fetch_add(BLOCK * 2, std::memory_order_relaxed); //x2 since win = 2
        if (cur->movesN == 0)
        { //leaf, update result
            int r = (orginalPlayer ^ *player) * 2;
            do
            {
                cur->wins.fetch_add(r * BLOCK, std::memory_order_relaxed);
                cur = cur->parent;
            } while (cur->parent);
            return nullptr;
        }

        //do move
        board[cur->moves[i][2]][cur->moves[i][3]] = board[cur->moves[i][0]][cur->moves[i][1]];
        board[cur->moves[i][0]][cur->moves[i][1]] = '.';
        if (cur->moves[i][0] - cur->moves[i][2] == 2 || cur->moves[i][0] - cur->moves[i][2] == -2) //capturign move
            board[(cur->moves[i][2] + cur->moves[i][0]) / 2][(cur->moves[i][3] + cur->moves[i][1]) / 2] = '.';
        if (board[cur->moves[i][2]][cur->moves[i][3]] == 'l' && cur->moves[i][3] == 0) //light promotion
            board[cur->moves[i][2]][cur->moves[i][3]] = 'L';
        if (board[cur->moves[i][2]][cur->moves[i][3]] == 'd' && cur->moves[i][3] == TBoardSize - 1) //dark promotion
            board[cur->moves[i][2]][cur->moves[i][3]] = 'D';

        cur = cur->children[i];
        if (cur->position == -1)
            *player = *player ^ 1;
    }
    cur->games.fetch_add(BLOCK * 2, std::memory_order_relaxed);
    if (cur->movesN == -1)
    {
        Expand<TRules, TBoardSize, TMaxMoves>(cur, board, *player);
    }
    return cur;
}

__global__ void initCurand(curandState* states)
{
    int idx = threadIdx.x;
    curand_init(clock64(), idx, 0, &states[idx]);
    //curand_init(idx, idx, 0, &states[idx]);
}

template <class TRules, int TBoardSize, int TMaxMoves>
void* thread(void* data)
{
    const int boardArea = TBoardSize * TBoardSize;

    ThreadData<TBoardSize>* d = (ThreadData<TBoardSize>*)data;
    TreeNode* cur = nullptr;
    char b[TBoardSize][TBoardSize];
    int p;

    initCurand<<<1, BLOCK, 0, d->stream>>>(d->states);
    cudaStreamSynchronize(d->stream);
    while (running)
    {
        cur = nullptr;

        //critical part - tree search
        pthread_mutex_lock(d->rootMutex);
        memcpy(b, d->board, boardArea * sizeof(char));
        p = d->player;

        while (cur == nullptr || cur->movesN == 0)
            cur = MCTSSelectionAndExpansion<TRules, TBoardSize, TMaxMoves>(b, d->root, &p);
        pthread_mutex_unlock(d->rootMutex);
        //critical part end

        int n = cur->movesN;
        for (int i = 0; i < BLOCK; i++)
        {
            memcpy(d->boards + (i * boardArea * sizeof(char)), b, boardArea * sizeof(char));
            d->positions[i] = cur->children[i % n]->position;
        }

        cudaMemPrefetchAsync(d->boards, BLOCK * sizeof(char[TBoardSize][TBoardSize]), device, d->stream);
        cudaMemPrefetchAsync(d->positions, BLOCK * sizeof(int), device, d->stream);
        cudaMemPrefetchAsync(d->results, BLOCK * sizeof(float), device, d->stream);
        MCTSSimulation<TRules, TBoardSize, TMaxMoves><<<1, BLOCK, 0, d->stream>>>(d->boards, d->positions, d->states, p, Player, d->results);
        cudaStreamSynchronize(d->stream);
        int s = 0;
        for (int i = 0; i < BLOCK; i++)
        {
            int r = d->results[i];
            cur->children[i % n]->games.fetch_add(2, std::memory_order_relaxed);
            if (r)
            {
                s += r;
                cur->children[i % n]->wins.fetch_add(r, std::memory_order_relaxed);
            }
        }
        while (cur->parent)
        {
            cur->wins.fetch_add(s, std::memory_order_relaxed);
            cur = cur->parent;
        }
        cur->wins.fetch_add(s, std::memory_order_relaxed);

        kernels.fetch_add(1, std::memory_order_relaxed);
        //end
    }
    return NULL;
}

// find best move for light player on given board using GPU
template <class TRules, int TBoardSize, int TMaxMoves>
void findMoveGPU(char board[TBoardSize][TBoardSize], int timeout, int player)
{
    const int boardArea = TBoardSize * TBoardSize;

    cudaGetDevice(&device);
    Player = player;
    float elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //cudaFuncSetCacheConfig(MCTSSimulation, cudaFuncCachePreferL1);
    cudaProfilerStart();

    TreeNode* root = new TreeNode(nullptr);
    Expand<TRules, TBoardSize, TMaxMoves>(root, board, player);

    pthread_t threads[THREADS];
    ThreadData<TBoardSize> tData[THREADS];
    pthread_mutex_t rootMutex;
    pthread_mutex_init(&rootMutex, NULL);

    for (int i = 0; i < THREADS; i++)
    {
        cudaMallocManaged(&tData[i].stream, sizeof(cudaStream_t));
        cudaStreamCreate(&tData[i].stream);
        cudaMallocManaged(&tData[i].boards, BLOCK * sizeof(char[TBoardSize][TBoardSize]));
        cudaMallocManaged(&tData[i].positions, BLOCK * sizeof(int));
        cudaMallocManaged(&tData[i].results, BLOCK * sizeof(float));
        cudaMallocManaged(&tData[i].states, BLOCK * sizeof(curandState));
        cudaMemAdvise(tData[i].states, BLOCK * sizeof(curandState), cudaMemAdviseSetPreferredLocation, device);
        tData[i].root = root;
        tData[i].rootMutex = &rootMutex;
        tData[i].player = player;
        memcpy(tData[i].board, board, boardArea * sizeof(char));
        pthread_create(&threads[i], NULL, thread<TRules, TBoardSize, TMaxMoves>, (void*)&tData[i]);
    }

    while (elapsed < timeout)
    {
        usleep(10000); //10ms
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
    }
    running = false;
    cudaProfilerStop();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < THREADS; i++)
        pthread_join(threads[i], NULL);

    TreeNode* cur = root;
    int movesCount = 0;
    do
    {
        int best = getBestMove(cur);
        movesCount++;
        cur = cur->children[best];
    } while (cur->position != -1);

//    printf("%d\n", movesCount);

    cur = root;
    do
    {
        int best = getBestMove(cur);
 //       printf("%d %d %d %d\n", cur->moves[best][0], cur->moves[best][1], cur->moves[best][2], cur->moves[best][3]);
        board[cur->moves[best][2]][cur->moves[best][3]] = board[cur->moves[best][0]][cur->moves[best][1]];
        board[cur->moves[best][0]][cur->moves[best][1]] = '.';
        if (cur->moves[best][0] - cur->moves[best][2] == 2 || cur->moves[best][0] - cur->moves[best][2] == -2) //capturign move
            board[(cur->moves[best][2] + cur->moves[best][0]) / 2][(cur->moves[best][3] + cur->moves[best][1]) / 2] = '.';
        if (board[cur->moves[best][2]][cur->moves[best][3]] == 'l' && cur->moves[best][3] == 0) //light promotion
            board[cur->moves[best][2]][cur->moves[best][3]] = 'L';
        if (board[cur->moves[best][2]][cur->moves[best][3]] == 'd' && cur->moves[best][3] == TBoardSize - 1) //dark promotion
            board[cur->moves[best][2]][cur->moves[best][3]] = 'D';
        cur = cur->children[best];
    } while (cur->position != -1);
//    printf("W:%d G:%d K:%d\n", root->wins.load() / 2, root->games.load() / 2, kernels.load());
    for (int i = 0; i < THREADS; i++)
    {
        cudaStreamDestroy(tData[i].stream);
        cudaFree(&tData[i].states);
        cudaFree(&tData[i].stream);
        cudaFree(&tData[i].boards);
        cudaFree(&tData[i].positions);
        cudaFree(&tData[i].results);
    }

    FreeMemory(root);
}
