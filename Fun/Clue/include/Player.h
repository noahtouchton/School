#pragma once
#include <string>
#include "Constants.h"
#include "LinkedList.h"
#include <iostream>
#include "Log.h"



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
        void takeTurn();
        void learnCard(int card);
        void addCard(int card);

        std::string getName();
};