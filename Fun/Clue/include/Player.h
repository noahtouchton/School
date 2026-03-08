#pragma once
#include <string>
#include "Constants.h"
#include "LinkedList.h"
#include <iostream>



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
        // Constructor declaration
        Player();
        Player(std::string playerName, bool computer);

        // Method declaration
        void takeTurn();
        void learnCard(int card);
        void addCard(int card);
};