#include <iostream>
#include <cstdlib>
#include <ctime>
#include <limits>
#include "../include/Game.h"

bool globalAutoSkip = false;

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    std::cout << std::unitbuf;
    srand(static_cast<unsigned int>(time(nullptr)));
    int numPlayers, numCOMPlayers, totalPlayers;
    std::cout << "Welcome to Clue!" << std::endl;
    int debugInput = 0;
    std::cout << "Enable COM debugging mode? (1 for yes, 0 for no): ";
    std::cin >> debugInput;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    comDebugMode = (debugInput == 1);
    do {
        std::cout << "Enter the number of human players (0-6): ";
        std::cin >> numPlayers;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
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
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            totalPlayers = numPlayers + numCOMPlayers;
        } while (totalPlayers < 0 || totalPlayers > 6);
    }
    std::cout << "Starting game with " << numPlayers << " human players and " << numCOMPlayers << " computer players." << std::endl;
    globalAutoSkip = (numPlayers == 0);
    Game game(numPlayers, numCOMPlayers);
    game.loop();

    return 0;
}