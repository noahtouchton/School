#pragma once
#include "Player.h"
#include "Constants.h"
#include "Board.h"


class Game {
    private:
        // Game Constants
        int numPlayers;
        int numCOMPlayers;
        const int MAX_PLAYERS = 6;
        int secretSuspect;
        int secretWeapon;
        int secretRoom;
        Player* players;
        // Game State Variables
        int round;
        int currentPlayer;
        bool gameOver;
        Board board;

        void initGame();
        void selectSecretCards();
        void dealCards();



    public:
        // Constructor declaration
        Game(int players, int comPlayers);

        // Method declaration
        void loop();

        void takeTurn();

};