#pragma once
#include "Log.h"


class Board {
    private:
        int playerCount;
        int* calculateSubIndex(int counter){
            int* subIndex = new int[2];
            subIndex[0] = counter / 3;
            subIndex[1] = counter % 3;
            return subIndex;
        }
    public:
        int* positions;
        Logger logger;

        Board() : logger(LogLevel::Game), playerCount(0) {
            positions = nullptr;
        }
        Board(int numPlayers, LogLevel logLevel) : logger(logLevel), playerCount(numPlayers) {
            positions = new int[numPlayers];
            for (int i = 0; i < numPlayers; i++) {
                positions[i] = 4; // Starting position for each player
            }
        }

        ~Board() {
            delete[] positions;
        }

        void movePlayer(int playerIndex, int newPos) {
            if (playerIndex >= 0 && playerIndex < playerCount) {
                positions[playerIndex] = newPos;
            }
        }

        void printBoard() {
            logger.print(LogLevel::Game, "Player Positions:");

            char board[8][8];
            for(int i=0; i<8; i++) {
                for(int j=0; j<8; j++) {
                    board[i][j] = '*';
                }
            }
            for(int i=0; i<3; i++) {
                for(int j=0; j<3; j++) {
                    int pos = i*3 +j;
                    int counter = 0;
                    for(int k=0; k<sizeof(positions);k++) {
                        if(positions[k] == pos) {
                            int* subIndex = calculateSubIndex(counter);
                            board[3*i+subIndex[0]][3*j+subIndex[1]] = char(1 + k); // Mark player position with player number
                            delete[] subIndex;
                            counter++;
                        }
                    }
                }
            }
            logger.printBoard(board); 
        }
};

// ***|***|***
// ***|***|***
// ***|***|***
// ---|---|---
// ***|***|***
// ***|***|***
// ***|***|***
// ---|---|---|
// ***|***|***
// ***|***|***
// ***|***|***