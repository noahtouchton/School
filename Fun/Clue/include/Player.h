#pragma once
#include <string>
#include "Constants.h"
#include "LinkedList.h"
#include <iostream>
#include "Log.h"

class Game;
class Board;

class Player {
    private:
        int position;
        bool isComputer;
        int numCards;
        std::string name;
        LinkedList* cards;

        LinkedList* susPeople;
        LinkedList* susWeapons;
        LinkedList* susRooms;

        
        int cardToNum(std::string card, int type);
        std::string numToCard(int num, int type);
        LinkedList* loadList(int len);

    public:
        Logger logger;
        // Constructor declaration
        Player();
        Player(std::string playerName, bool computer, LogLevel logLevel);

        // Method declaration
        void takeTurn(Game& game, Board& board, int currentPlayer);
        void learnCard(int card);
        void addCard(int card);

        void movePlayer(int roll, Board& board, int currentPlayer);
        Suggestion makeSuggestion();

        std::string getName();
};