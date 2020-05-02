#include <winsock2.h>
#include <windows.h>
#include <stdio.h>
#pragma warning(disable : 4996)

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

//structs and defines from http://www.fierz.ch/cbdeveloper.php
struct coor
{
	int x;
	int y;
};
struct CBmove
{
	int jumps; // number of jumps in this move
	int newpiece; // moving piece after jump
	int oldpiece; // moving piece before jump
	struct coor from, to; // from,to squares of moving piece
	struct coor path[12]; // intermediate squares to jump to
	struct coor del[12]; // squares where men are removed
	int delpiece[12]; // piece type which is removed
};
#define WHITE 1
#define BLACK 2
#define MAN 4
#define KING 8
#define FREE 0

int WINAPI enginecommand(char command[256], char reply[1024]) 
{
	if (strncmp(command, "name", 4) == 0) {
		strncpy(reply, "checkers-mc client", 256);
		return 1;
	}
	if (strncmp(command, "about", 5) == 0) {
		strncpy(reply, "client for https://github.com/franekjel/checkers-mc", 256);
		return 1;
	}
	strncpy(reply, "?", 256);
	return 0;
}

int WINAPI getmove(int board[8][8], int color, double maxtime, char str[1024], int *playnow, int info, int moreinfo, struct CBmove *move) 
{
	char b[8][8];
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			b[i][7 - j] = '.';
			if (board[j][i] == (BLACK | MAN))
				b[i][7 - j] = 'd';
			if (board[j][i] == (BLACK | KING))
				b[i][7 - j] = 'D';
			if (board[j][i] == (WHITE | MAN))
				b[i][7 - j] = 'l';
			if (board[j][i] == (WHITE | KING))
				b[i][7 - j] = 'L';
		}
	}

	WSADATA wsaData;
	int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (result != NO_ERROR) {
		sprintf_s(str, 256, "WSAStartup error %d", result);
		return 3;
	}
	SOCKET server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (server == INVALID_SOCKET)
	{
		sprintf_s(str, 256, "Error creating socket: %ld", WSAGetLastError());
		WSACleanup();
		return 3;
	}

	SOCKADDR_IN addr;
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	strncpy(str, "connecting", 256);
	addr.sin_addr.s_addr = inet_addr("192.168.1.80");//TODO: make it parameter ot at least define
	addr.sin_port = htons(11821);//TODO: parameter
	connect(server, (SOCKADDR *)&addr, sizeof(addr));
	char buffer[80];
	strncpy(str, "connected", 256);
	//message is 0 or 1 on first byte (player) then 64 bytes of board then \n\0
	buffer[0] = '0'+ (color - 1);//checkers-mc use 0 for white and 1 for black
	for (int i = 0; i < 64; i++)
		buffer[i+1] = b[i / 8][i % 8];
	buffer[65] = '\n';
	buffer[66] = 0;
	int size = 67;
	int NSize = htonl(size);
	send(server, &NSize, 4, 0);//we send size of message (it will be always 67 but size on the begining is convenient)
	int sended = 0;
	while (sended < size) {//then we send message
		sended+= send(server, buffer+sended, size-sended, 0);
	}
	recv(server, &NSize, 4, 0);//we wait to receive size of message
	size = ntohl(NSize);
	int received = 0;
	while (received < size) {//then we receive full message
		received += recv(server, buffer + received, size - received, 0);
		sprintf_s(str, 256, "received %d\\%d bytes", received,size);
	}

	//return message contains \n after each 8 squares
	int j = 0;
	for (int i = 0; i < size; i++) {
		if(buffer[i]=='.' || buffer[i] == 'l' || buffer[i] == 'L' || buffer[i] == 'd' || buffer[i] == 'D'){
			b[j / 8][j % 8]=buffer[i];
			j++;
		}
	}
	
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			board[7-i][j] = FREE;
			if (b[j][i] == 'd')
				board[7-i][j] = BLACK | MAN;
			if (b[j][i] == 'D')
				board[7-i][j] = BLACK | KING;
			if (b[j][i] == 'l')
				board[7-i][j] = WHITE | MAN;
			if (b[j][i] == 'L')
				board[7-i][j] = WHITE | KING;
		}
	}
	closesocket(server);

	return 3;//UNKNOWN
}
