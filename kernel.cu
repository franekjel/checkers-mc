#include "kernel.h"

int Player = 0;

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

struct ThreadData
{
    char board[8][8];
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

static void Expand(TreeNode* node, char board[8][8], int player)
{
    Move moves[48];
    int captures = 0, n = 0;
    if (node->position != -1)
    {
        if (player)
            getMovesDarkPos(board, moves, &captures, (node->position) % 8, (node->position) / 8, n);
        else
            getMovesLightPos(board, moves, &captures, (node->position) % 8, (node->position) / 8, n);
    } else
    {
        if (player) //0-light 1-dark
            n = genMovesDark(board, moves, &captures);
        else
            n = genMovesLight(board, moves, &captures);
    }
    if (captures > 0)
        n = filterMoves(moves, n);
    node->movesN = n;
    node->children = new TreeNode*[n];
    node->moves = new Move[n];
    for (int i = 0; i < n; i++)
    {
        memcpy(node->moves[i], moves[i], sizeof(Move));
        node->children[i] = new TreeNode(node);
        if (captures > 0)
        { //if there are capturing moves all are capturing. So we check if after capture we have another possibility to capture
            char b[8][8];
            memcpy(b, board, 64 * sizeof(char));
            b[node->moves[i][2]][node->moves[i][3]] = b[node->moves[i][0]][node->moves[i][1]];
            b[node->moves[i][0]][node->moves[i][1]] = '.';
            b[(node->moves[i][0] + node->moves[i][2]) / 2][(node->moves[i][1] + node->moves[i][3]) / 2] = '.';
            int m = 0;
            Move mvs[4];
            int cptr = 0;
            if (player)
                getMovesDarkPos(b, mvs, &cptr, node->moves[i][2], node->moves[i][3], m);
            else
                getMovesLightPos(b, mvs, &cptr, node->moves[i][2], node->moves[i][3], m);
            if (cptr > 0)
            {
                node->children[i]->position = node->moves[i][2] + 8 * node->moves[i][3];
            }
        }
    }
}

//simulation starts in node with children, we choose one and do random game. Then backpropagate result
//__global__ void __launch_bounds__(BLOCK) MCTSSimulation(char* boards, int* positions, curandState* states, int player, int orginalPlayer, float* results)
__global__ void MCTSSimulation(char* boards, int* positions, curandState* states, int player, int orginalPlayer, float* results)
{
    int idx = threadIdx.x;
    curandState state = states[idx];
    char board[8][8];
    memcpy(board, boards + (64 * idx), 64 * sizeof(char));
    int dep = 0; //depth
    float r = 0; //result
    Move moves[48];
    int captures = 0;
    int xpos = positions[idx] % 8, ypos = positions[idx] / 8;
    positions[idx] = -1;
    while (dep < 120)
    { //NOTE: magic number! - max depth of simulation
        int n = 0;
        captures = 0;
        if (xpos > -1)
        {
            if (player)
                getMovesDarkPos(board, moves, &captures, xpos, ypos, n);
            else
                getMovesLightPos(board, moves, &captures, xpos, ypos, n);
        } else
        {
            if (player)
                n = genMovesDark(board, moves, &captures);
            else
                n = genMovesLight(board, moves, &captures);
        }
        if (n == 0)
        {
            r = player ^ orginalPlayer;
            r *= 2;
            break;
        }

        if (captures > 0)
            n = filterMoves(moves, n);
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
                getMovesDarkPos(board, moves, &cptr, xpos, ypos, m);
            else
                getMovesLightPos(board, moves, &cptr, xpos, ypos, m);
            m = filterMoves(moves, m);
            if (m == 0)
            {
                xpos = -1;
                player = player ^ 1;
            }
        } else
            player = player ^ 1;
        int c = checkResult(board);
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

static TreeNode* MCTSSelectionAndExpansion(char board[8][8], TreeNode* root, int* player)
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
        if (board[cur->moves[i][2]][cur->moves[i][3]] == 'd' && cur->moves[i][3] == 7) //dark promotion
            board[cur->moves[i][2]][cur->moves[i][3]] = 'D';

        cur = cur->children[i];
        if (cur->position == -1)
            *player = *player ^ 1;
    }
    cur->games.fetch_add(BLOCK * 2, std::memory_order_relaxed);
    if (cur->movesN == -1)
    {
        Expand(cur, board, *player);
    }
    return cur;
}

__global__ void initCurand(curandState* states)
{
    int idx = threadIdx.x;
    curand_init(clock64(), idx, 0, &states[idx]);
}

void* thread(void* data)
{
    ThreadData* d = (ThreadData*)data;
    TreeNode* cur = nullptr;
    char b[8][8];
    int p;

    initCurand<<<1, BLOCK, 0, d->stream>>>(d->states);
    cudaStreamSynchronize(d->stream);

    while (running)
    {
        cur = nullptr;
        //critical part - tree search
        pthread_mutex_lock(d->rootMutex);
        memcpy(b, d->board, 64 * sizeof(char));
        p = d->player;
        while (cur == nullptr || cur->movesN == 0)
            cur = MCTSSelectionAndExpansion(b, d->root, &p);
        pthread_mutex_unlock(d->rootMutex);
        //critical part end
        int n = cur->movesN;
        for (int i = 0; i < BLOCK; i++)
        {
            memcpy(d->boards + (i * 64 * sizeof(char)), b, 64 * sizeof(char));
            d->positions[i] = cur->children[i % n]->position;
        }
        MCTSSimulation<<<1, BLOCK, 0, d->stream>>>(d->boards, d->positions, d->states, p, Player, d->results);
        cudaStreamSynchronize(d->stream);
        int s = 0;
        for (int i = 0; i < BLOCK; i++)
        {
            int r = d->results[i];
            cur->children[d->positions[i]]->games.fetch_add(2, std::memory_order_relaxed);
            if (r)
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
        cur->wins.fetch_add(s, std::memory_order_relaxed); //

        //end
    }
    return NULL;
}

void findMoveGPU(char board[8][8], int timeout, int player)
{
    Player = player;
    float elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaFuncSetCacheConfig(MCTSSimulation, cudaFuncCachePreferL1);
    cudaProfilerStart();

    TreeNode* root = new TreeNode(nullptr);
    Expand(root, board, player);

    pthread_t threads[THREADS];
    ThreadData tData[THREADS];
    pthread_mutex_t rootMutex;
    pthread_mutex_init(&rootMutex, NULL);

    for (int i = 0; i < THREADS; i++)
    {
        cudaMallocManaged(&tData[i].stream, sizeof(cudaStream_t));
        cudaStreamCreate(&tData[i].stream);
        cudaMallocManaged(&tData[i].boards, BLOCK * sizeof(char[8][8]));
        cudaMallocManaged(&tData[i].positions, BLOCK * sizeof(int));
        cudaMallocManaged(&tData[i].results, BLOCK * sizeof(float));
        cudaMallocManaged(&tData[i].states, BLOCK * sizeof(curandState));
        tData[i].root = root;
        tData[i].rootMutex = &rootMutex;
        tData[i].player = player;
        memcpy(tData[i].board, board, 64 * sizeof(char));
        pthread_create(&threads[i], NULL, thread, (void*)&tData[i]);
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

    //check and do best  move
    float max = 0;
    int maxi = 0;
    for (int i = 0; i < root->movesN; i++)
    {
        float g = root->children[i]->games.load();
        float w = root->children[i]->wins.load();
        printf("%f\n", w / g);
        if (w / g > max)
        {
            max = w / g;
            maxi = i;
        }
    }
    int i = maxi;
    TreeNode* cur = root;
    do
    {
        board[cur->moves[i][2]][cur->moves[i][3]] = board[cur->moves[i][0]][cur->moves[i][1]];
        board[cur->moves[i][0]][cur->moves[i][1]] = '.';
        if (cur->moves[i][0] - cur->moves[i][2] == 2 || cur->moves[i][0] - cur->moves[i][2] == -2) //capturign move
            board[(cur->moves[i][2] + cur->moves[i][0]) / 2][(cur->moves[i][3] + cur->moves[i][1]) / 2] = '.';
        if (board[cur->moves[i][2]][cur->moves[i][3]] == 'l' && cur->moves[i][3] == 0) //light promotion
            board[cur->moves[i][2]][cur->moves[i][3]] = 'L';
        if (board[cur->moves[i][2]][cur->moves[i][3]] == 'd' && cur->moves[i][3] == 7) //dark promotion
            board[cur->moves[i][2]][cur->moves[i][3]] = 'D';
        cur = cur->children[i];
    } while (cur->position != -1);
    printf("W:%d G:%d", root->wins.load(), root->games.load());
    for (int i = 0; i < THREADS; i++)
    {
        cudaStreamDestroy(tData[i].stream);
        cudaFree(&tData[i].states);
        cudaFree(&tData[i].states);
        cudaFree(&tData[i].stream);
        cudaFree(&tData[i].boards);
        cudaFree(&tData[i].positions);
        cudaFree(&tData[i].results);
    }

    FreeMemory(root);
}
