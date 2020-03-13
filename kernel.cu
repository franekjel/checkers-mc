#include "kernel.h"

int Player = 0;

volatile sig_atomic_t running = 1;

Rules* hostRules = new AmericanRules();
__device__ Rules* deviceRules;

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
    char* board;
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

static void Expand(TreeNode* node, char* board, int player)
{
    int N = hostRules->boardSize;
    int N2 = N * N;
    Move moves[hostRules->maxMoves];
    int captures = 0, n = 0;
    if (node->position != -1)
    {
        if (player)
            hostRules->getMovesDarkPos(board, moves, captures, (node->position) % 8, (node->position) / 8, n);
        else
            hostRules->getMovesLightPos(board, moves, captures, (node->position) % 8, (node->position) / 8, n);
    } else
    {
        if (player) //0-light 1-dark
            n = hostRules->genMovesDark(board, moves, captures);
        else
            n = hostRules->genMovesLight(board, moves, captures);
    }
    if (captures > 0)
        n = hostRules->filterMoves(moves, n);
    node->movesN = n;
    node->children = new TreeNode*[n];
    node->moves = new Move[n];
    for (int i = 0; i < n; i++)
    {
        memcpy(node->moves[i], moves[i], sizeof(Move));
        node->children[i] = new TreeNode(node);
        if (captures > 0)
        { //if there are capturing moves all are capturing. So we check if after capture we have another possibility to capture
            char b[N2];
            memcpy(b, board, N2 * sizeof(char));
            int x1 = node->moves[i][0], y1 = node->moves[i][1], x2 = node->moves[i][2], y2 = node->moves[i][3];
            b[y2 * N + x2] = b[y1 * N + x1];
            b[y2 * N + x2] = b[y1 * N + x1] = '.';
            if (x1 - x2 == 2 || x1 - x2 == -2)
                b[(y1 + y2) * (N / 2) + (x1 + x2) / 2] = '.';
            int m = 0;
            Move mvs[hostRules->maxMoves];
            int cptr = 0;
            if (player)
                hostRules->getMovesDarkPos(b, mvs, cptr, x2, y2, m);
            else
                hostRules->getMovesLightPos(b, mvs, cptr, x2, y2, m);
            if (cptr > 0)
            {
                node->children[i]->position = N * y2 + x2;
            }
        }
    }
}

//simulation starts in node with children, we choose one and do random game. Then backpropagate result
__global__ void MCTSSimulation(char* boards, int* positions, curandState* states, int player, int originalPlayer, float* results)
{
    int N = deviceRules->boardSize;
    int N2 = deviceRules->boardSize * deviceRules->boardSize;
    int idx = threadIdx.x;
    curandState state = states[idx];
    char* board = boards + (N2 * idx);
    //memcpy(board, boards + (N2 * idx), N2 * sizeof(char));
    int dep = 0; //depth
    float r = 0; //result
    Move moves[48]; //TODO something, cudamalloc etc
    int captures = 0;
    int xpos = positions[idx] % N, ypos = positions[idx] / N;
    positions[idx] = -1;
    while (dep < 120)
    { //NOTE: magic number! - max depth of simulation
        int n = 0;
        captures = 0;
        if (xpos > -1)
        {
            if (player)
                deviceRules->getMovesDarkPos(board, moves, captures, xpos, ypos, n);
            else
                deviceRules->getMovesLightPos(board, moves, captures, xpos, ypos, n);
        } else
        {
            if (player)
                n = deviceRules->genMovesDark(board, moves, captures);
            else
                n = deviceRules->genMovesLight(board, moves, captures);
        }
        if (n == 0)
        {
            r = player ^ originalPlayer;
            r *= 2;
            break;
        }

        if (captures > 0)
            n = deviceRules->filterMoves(moves, n);
        int i = curand(&state) % n;
        if (dep == 0)
            positions[idx] = i;
        int x1 = moves[i][0], y1 = moves[i][1], x2 = moves[i][2], y2 = moves[i][3];
        board[y2 * N + x2] = board[y1 * N + x1];
        board[y1 * N + x1] = '.';
        if (x1 - x2 == 2 || x1 - x2 == -2)
        { //capturign move
            board[(y2 - y1) * (N / 2) + (x2 - x1) / 2] = '.';
            xpos = x2;
            ypos = y2;
            int cptr = 0, m = 0;
            if (player)
                deviceRules->getMovesDarkPos(board, moves, cptr, xpos, ypos, m);
            else
                deviceRules->getMovesLightPos(board, moves, cptr, xpos, ypos, m);
            m = deviceRules->filterMoves(moves, m);
            if (m == 0)
            {
                xpos = -1;
                player = player ^ 1;
            }
        } else
            player = player ^ 1;
        int c = deviceRules->checkResult(board);
        if (c != -1)
        {
            if (c == originalPlayer)
                r = 2;
            break;
        }
        dep++;
    } //simulation
    if (dep == 120)
        r = 1;

    results[idx] = r;
}

static TreeNode* MCTSSelectionAndExpansion(char* board, TreeNode* root, int* player)
{
    int N = hostRules->boardSize;
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
        int x1 = cur->moves[i][0], y1 = cur->moves[i][1], x2 = cur->moves[i][2], y2 = cur->moves[i][3];

        //do move
        board[y2 * N + x2] = board[y1 * N + x1];
        board[y1 * N + x1] = '.';
        if (x1 - x2 == 2 || x1 - x2 == -2) //capturign move
            board[(y1 + y2) * (N / 2) + ((x1 + x2) / 2)] = '.'; //*(N/2) because (y1+y2)/2 * N
        if (board[y2 * N + x2] == 'l' && y2 == 0) //light promotion
            board[y2 * N + x2] = 'L';
        if (board[y2 * N + x2] == 'd' && y2 == N - 1) //dark promotion
            board[y2 * N + x2] = 'D';

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

__global__ void initAmericanRules()
{
    deviceRules = new AmericanRules();
}

void* thread(void* data)
{
    ThreadData* d = (ThreadData*)data;
    TreeNode* cur = nullptr;
    int N2 = hostRules->boardSize * hostRules->boardSize;
    char b[N2];
    int p;

    initCurand<<<1, BLOCK, 0, d->stream>>>(d->states);
    cudaStreamSynchronize(d->stream);

    while (running)
    {
        cur = nullptr;
        //critical part - tree search
        pthread_mutex_lock(d->rootMutex);
        memcpy(b, d->board, N2 * sizeof(char));
        p = d->player;
        while (cur == nullptr || cur->movesN == 0)
            cur = MCTSSelectionAndExpansion(b, d->root, &p);
        pthread_mutex_unlock(d->rootMutex);
        //end critical part
        int n = cur->movesN;
        for (int i = 0; i < BLOCK; i++)
        {
            memcpy(d->boards + (i * N2 * sizeof(char)), b, N2 * sizeof(char));
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
        cur->wins.fetch_add(s, std::memory_order_relaxed);
    }
    return NULL;
}

void findMoveGPU(char* board, int timeout, int player, RulesType rules)
{
    switch (rules)
    {
    case AMERICAN_RULES:
        hostRules = new AmericanRules();
        initAmericanRules<<<1, 1, 0>>>();
        break;
    }

    int N = hostRules->boardSize;

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
        cudaMallocManaged(&tData[i].boards, BLOCK * sizeof(hostRules->boardSize));
        cudaMallocManaged(&tData[i].positions, BLOCK * sizeof(int));
        cudaMallocManaged(&tData[i].results, BLOCK * sizeof(float));
        cudaMallocManaged(&tData[i].states, BLOCK * sizeof(curandState));
        tData[i].root = root;
        tData[i].rootMutex = &rootMutex;
        tData[i].player = player;
        memcpy(tData[i].board, board, hostRules->boardSize * sizeof(char));
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

    int x1 = cur->moves[i][0], y1 = cur->moves[i][1], x2 = cur->moves[i][2], y2 = cur->moves[i][3];
    do
    {
        board[y2 * N + x2] = board[y1 + N + x1];
        board[y1 + N + x1] = '.';
        if (x1 - x2 == 2 || y1 - y2 == -2) //capturign move
            board[(y1 + y2) * (N / 2) + (x1 + x2) / 2] = '.';
        if (board[y2 * N + x2] == 'l' && y2 == 0) //light promotion
            board[y2 * N + x2] = 'L';
        if (board[y2 * N + x2] == 'd' && y2 == N - 1) //dark promotion
            board[y2 * N + x2] = 'D';
        cur = cur->children[i];
    } while (cur->position != -1);

    delete hostRules;
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
