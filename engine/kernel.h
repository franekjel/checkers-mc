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
    int* results;
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

static int chooseNode(TreeNode* cur, int player)
{
    float max = -2;
    int maxi = 0;
    float N0 = cur->games.load();
    for (int i = 0; i < cur->movesN; i++)
    {
        float g = cur->children[i]->games.load();
        if (g == 0)
            return i;
        else
        {
            float w = cur->children[i]->wins.load();
            float u = w / g;
            if (player != Player)
                u *= -1;
            u += 0.7 * sqrtf(logf(N0) / g);
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
    float max = -1;
    int maxi = -1;
    for (int i = 0; i < cur->movesN; i++)
    {
        float g = cur->children[i]->games.load();
        if (g > max)
        {
            max = g;
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
            bool promotion = TRules::doMove(b, node->moves[i]);
            if (promotion) //pawn after promotion cannot capture
                continue;
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
__global__ void MCTSSimulation(char* boards, int* positions, curandState* states, int player, int orginalPlayer, int* results)
{
    const int boardArea = TBoardSize * TBoardSize;
    int idx = threadIdx.x;
    curandState state = states[idx];
    char board[TBoardSize][TBoardSize];
    memcpy(board, boards + (boardArea * idx), boardArea * sizeof(char));
    int dep = 0; //depth
    int r = 1; //result - default draw
    Move moves[TMaxMoves];
    int captures = 0;
    int xpos = positions[idx] % TBoardSize, ypos = positions[idx] / TBoardSize;
    while (dep < 120)
    { //NOTE: magic number! - max depth of simulation
        int n = 0;
        captures = 0;
        if (xpos != -1)
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
        if (n == 0) //player cannot do any move
        {
            r = player ^ orginalPlayer; //if root player cannot move this is loss. Otherwise this is win.
            r *= 2;
            break;
        }
        if (captures > 0)
            n = filterMoves<TMaxMoves>(moves, n);
        int i = curand(&state) % n;
        if (dep == 0)
            positions[idx] = i; //here positions is to tell main thread to which branch thread go
        bool promotion = TRules::doMove(board, moves[i]);
        if (!promotion && (moves[i][0] - moves[i][2] == 2 || moves[i][0] - moves[i][2] == -2)) //capturign move and not promotion
        {
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
        {
            player = player ^ 1;
        }
        dep++;
    }

    results[idx] = r;
}

template <class TRules, int TBoardSize, int TMaxMoves>
static TreeNode* MCTSSelectionAndExpansion(char board[TBoardSize][TBoardSize], TreeNode* root, int* player, int orginalPlayer)
{
    int N0 = 20; //NOTE: magic number! - how fast node expands. Node is expanded after there is more than N0 games
    TreeNode* cur = root;
    while (cur->movesN != -1 && cur->games.load() > N0)
    {
        int i = chooseNode(cur, *player);
        cur->games.fetch_add(BLOCK * 2, std::memory_order_relaxed); //x2 since win = 2
        if (cur->movesN == 0)
        { //leaf, update result
            TreeNode* ret = cur;
            int r = (*player ^ orginalPlayer) * 2;
            do
            {
                cur->wins.fetch_add(r * BLOCK, std::memory_order_relaxed);
                cur = cur->parent;
            } while (cur->parent);
            return ret;
        }
        TRules::doMove(board, cur->moves[i]);
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
        memcpy(b, d->board, boardArea * sizeof(char));
        p = d->player;
        int i = 0;
        //critical part - tree search
        pthread_mutex_lock(d->rootMutex);
        while (cur == nullptr || cur->movesN == 0)
        {
            p = d->player;
            cur = MCTSSelectionAndExpansion<TRules, TBoardSize, TMaxMoves>(b, d->root, &p, Player);
            i++;
            if (!running || i > 10)
            {
                running = false;
                pthread_mutex_unlock(d->rootMutex);
                return NULL;
            }
        }
        pthread_mutex_unlock(d->rootMutex);
        //critical part end

        for (int i = 0; i < BLOCK; i++)
        {
            memcpy(d->boards + (i * boardArea * sizeof(char)), b, boardArea * sizeof(char));
            d->positions[i] = cur->position;
        }

        cudaMemPrefetchAsync(d->boards, BLOCK * sizeof(char[TBoardSize][TBoardSize]), device, d->stream);
        cudaMemPrefetchAsync(d->positions, BLOCK * sizeof(int), device, d->stream);
        cudaMemPrefetchAsync(d->results, BLOCK * sizeof(float), device, d->stream);
        MCTSSimulation<TRules, TBoardSize, TMaxMoves><<<1, BLOCK, 0, d->stream>>>(d->boards, d->positions, d->states, p, Player, d->results);
        cudaStreamSynchronize(d->stream);
        cudaMemPrefetchAsync(d->results, BLOCK * sizeof(float), cudaCpuDeviceId, d->stream);
        cudaMemPrefetchAsync(d->positions, BLOCK * sizeof(int), cudaCpuDeviceId, d->stream);
        cudaStreamSynchronize(d->stream);
        int s = 0;
        for (int i = 0; i < BLOCK; i++)
        {
            int r = d->results[i];
            cur->children[d->positions[i]]->games.fetch_add(2, std::memory_order_relaxed);
            if (r != 0)
            {
                s += r;
                cur->children[d->positions[i]]->wins.fetch_add(r, std::memory_order_relaxed);
            }
        }
        while (cur->parent)
        {
            cur->wins.fetch_add(s, std::memory_order_relaxed);
            cur = cur->parent;
        }
        cur->wins.fetch_add(s, std::memory_order_relaxed);

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
        cudaMallocManaged(&tData[i].results, BLOCK * sizeof(int));
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
    int movesCount = 1;
    while (cur->position != -1 || cur->movesN < 1)
    {
        int best = getBestMove(cur);
        cur = cur->children[best];
        movesCount++;
    }

    //printf("%d\n", movesCount);

    cur = root;
    do
    {
        int best = getBestMove(cur);
        //printf("%d %d %d %d\n", cur->moves[best][0], cur->moves[best][1], cur->moves[best][2], cur->moves[best][3]);
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
    /*
    printf("W:%d G:%d\n", root->wins.load() / 2, root->games.load() / 2);
    for (int i = 0; i < root->movesN; i++)
        printf("W:%d G:%d WG:%f\n", root->children[i]->wins.load() / 2, root->children[i]->games.load() / 2, float(root->children[i]->wins.load()) / float(root->children[i]->games.load()));
    */
    for (int i = 0; i < THREADS; i++)
    {
        cudaStreamDestroy(tData[i].stream);
        cudaFree(&tData[i].states);
        cudaFree(&tData[i].boards);
        cudaFree(&tData[i].positions);
        cudaFree(&tData[i].results);
    }

    FreeMemory(root);
}
