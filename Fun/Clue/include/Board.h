#pragma once
#include "Log.h"
#include <cmath>


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

            char board[9][9];
            for(int i=0; i<9; i++) {
                for(int j=0; j<9; j++) {
                    board[i][j] = '*';
                }
            }
            for(int i=0; i<3; i++) {
                for(int j=0; j<3; j++) {
                    int pos = i*3 +j;
                    int counter = 0;
                    for(int k=0; k<playerCount;k++) {
                        if(positions[k] == pos) {
                            int* subIndex = calculateSubIndex(counter);
                            board[3*i+subIndex[0]][3*j+subIndex[1]] = char('1' + k); // Mark player position with player number
                            delete[] subIndex;
                            counter++;
                        }
                    }
                }
            }
            logger.printBoard(board); 
        }

        void printValidBoard(int roll, int currentPlayer) {
            int isPossible[9] = {0};
            int k=0;

            int current_x = positions[currentPlayer] / 3;
            int current_y = positions[currentPlayer] % 3;

            for(int i=0; i<3; i++) {
                for(int j=0; j<3; j++) {
                    double dis = sqrt(pow(current_x - i, 2) + pow(current_y - j, 2));
                    if (dis <= double(roll)) {
                        isPossible[k] = 1;
                    }
                    k++;
            }

            isPossible[positions[currentPlayer]] = 0; // Can't stay in the same place

            logger.printColoredBoard(isPossible);

        }
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