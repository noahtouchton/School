#include <iostream>
#include "../include/Game.h"


int main() {
    int numPlayers, numCOMPlayers, totalPlayers;
    std::cout << "Welcome to Clue!" << std::endl;
    do {
        std::cout << "Enter the number of human players (0-6): ";
        std::cin >> numPlayers;
    } while (numPlayers < 0 || numPlayers > 6);

    int minCOM, maxCOM;

    if (numPlayers == 0) {
        minCOM = 1;
        maxCOM = 6;
    } else {
        minCOM = 0;
        maxCOM = 6 - numPlayers;
    }
    if (numPlayers == 6) {
        std::cout << "No computer players allowed when there are 6 human players." << std::endl;
        numCOMPlayers = 0;
    } else {
        do {
            std::cout << "Enter the number of computer players (" << minCOM << "-" << maxCOM << "): ";
            std::cin >> numCOMPlayers;
            totalPlayers = numPlayers + numCOMPlayers;
        } while (totalPlayers < 0 || totalPlayers > 6);
    }
    std::cout << "Starting game with " << numPlayers << " human players and " << numCOMPlayers << " computer players." << std::endl;
    Game game(numPlayers, numCOMPlayers);
    game.loop();

    return 0;
}