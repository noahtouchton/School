#include <iostream>
#include <format>
#include "../include/Game.h"
#include "../include/Board.h"

// Things that must be done in main prior to creating a Game object:
// 1. Ask the user how many players (1-6)
// 2. Ask the user how many COM (total players must be between 1 and 6, inclusive)


Game::Game(int players, int comPlayers) 
: numPlayers(players), numCOMPlayers(comPlayers), currentPlayer(0), gameOver(false), round(1), board(players + comPlayers, LogLevel::Game), logger(LogLevel::COM) {
    // Run the logic to randomly select the secret suspect, weapon, and room
    selectSecretCards();
    initGame();
    dealCards();

}

void Game::initGame() {
    // Logic to initialize the game state, such as shuffling cards and setting up players
    // assumes number of players is less than 6 as this will be checked in main before creating the Game object
    players = new Player[numPlayers+numCOMPlayers];
    for (int i = 0; i < numPlayers; i++) {
        std::string name = "Player " + std::to_string(i+1);
        players[i] = Player(name, false, LogLevel::COM); //find a way to have a universal log level later
    }
    for (int i = numPlayers; i < numPlayers + numCOMPlayers; i++) {
        std::string name = "COM " + std::to_string(i - numPlayers + 1);
        players[i] = Player(name, true, LogLevel::COM);
    }
}

void Game::selectSecretCards() {
    // Logic to randomly select the secret suspect, weapon, and room
    secretSuspect = rand() % ClueData::SUSPECTS.size();
    secretWeapon = rand() % ClueData::WEAPONS.size();
    secretRoom = rand() % ClueData::ROOMS.size();
}

void Game::dealCards() {
    // Logic to deal the remaining cards to the players
    int totalPlayers = numPlayers + numCOMPlayers;

    LinkedList* deck = new LinkedList();
    for (int i=0; i < ClueData::SUSPECTS.size(); i++) {
        if (i != secretSuspect) {
            deck->add(i);
        }
    }
    //weapons start at 10
    for (int i=0; i < ClueData::WEAPONS.size(); i++) {
        if (i != secretWeapon) {
            deck->add(i + 10);
        }
    }
    //rooms start at 20
    for (int i=0; i < ClueData::ROOMS.size(); i++) {
        if (i != secretRoom) {
            deck->add(i + 20);
        }
    }
    deck->shuffle();

    // Deal cards to players

    while (deck->getSize() > 0) {
        for (int i = 0; i < totalPlayers && deck->getSize() > 0; i++) {
            int card = deck->pop();
            players[i].addCard(card);
        }
    }

}

void Game::loop() {
    // Main game loop that continues until the game is over
    logger.print(LogLevel::Game, "Welcome to Clue!");
    board.printBoard();

    while (!gameOver) {
        logger.print(LogLevel::Game, "Round " + std::to_string(round) + ": " + players[currentPlayer].getName() + "'s turn");
        players[currentPlayer].takeTurn(*this, board, currentPlayer);
        break;
    }

}

