// player.cpp
#include "Player.h"
#include "Game.h"
#include "Board.h"
#include "Log.h"

// Constructor Implementation
Player::Player() : name(""), isComputer(false), position(5), numCards(0) {
    cards = nullptr;
    susPeople = nullptr;
    susWeapons = nullptr;
    susRooms = nullptr;
}
Player::Player(std::string playerName, bool computer, LogLevel logLevel)
    : name(playerName), isComputer(computer), position(5), numCards(0), logger(logLevel) {
    cards = new LinkedList();
    susPeople = loadList(ClueData::SUSPECTS.size());
    susWeapons = loadList(ClueData::WEAPONS.size());
    susRooms = loadList(ClueData::ROOMS.size());
}

// Helper Method Implementations
int Player::cardToNum(std::string card, int type) {
    if (type == 0) { 
        for (size_t i = 0; i < ClueData::SUSPECTS.size(); i++) {
            if (ClueData::SUSPECTS[i] == card) return i;
        }
    } else if (type == 1) {
        for (size_t i = 0; i < ClueData::WEAPONS.size(); i++) {
            if (ClueData::WEAPONS[i] == card) return i;
        }
    } else if (type == 2) {
        for (size_t i = 0; i < ClueData::ROOMS.size(); i++) {
            if (ClueData::ROOMS[i] == card) return i;
        }
    }
    return -1; 
}

std::string Player::numToCard(int num, int type) {
    if (type == 0) {
        if (num >= 0 && num < ClueData::SUSPECTS.size()) return ClueData::SUSPECTS[num];
    } else if (type == 1) {
        if (num >= 0 && num < ClueData::WEAPONS.size()) return ClueData::WEAPONS[num];
    } else if (type == 2) {
        if (num >= 0 && num < ClueData::ROOMS.size()) return ClueData::ROOMS[num];
    }
    return ""; 
}

LinkedList* Player::loadList(int len) {
    LinkedList* list = new LinkedList();
    for (int i = 0; i < len; i++) {
        list->add(i);
    }
    return list;
}


// Gameplay Method Implementations
void Player::takeTurn(Game& game, Board& board, int currentPlayer) {
    LogLevel playerLevel = isComputer ? LogLevel::COM : LogLevel::Player;

    int roll = rand() % 3 + 1;
    logger.print(playerLevel, name + " rolled a " + std::to_string(roll));
    //Get new room position

    movePlayer(roll, board, currentPlayer);

    Suggestion suggestion;



}

void Player::learnCard(int card) {
    if (card%10 == 0) {
        susPeople->remove(card);
    } else if (card%10 == 1) {
        susWeapons->remove(card%10);
    } else if (card%10 == 2) {
        susRooms->remove(card%20);
    }
}

void Player::addCard(int card) {
    cards->add(card);
    learnCard(card);
    numCards++;
}

std::string Player::getName() {
    return name;
}

void Player::movePlayer(int roll, Board& board, int currentPlayer) {
    if (!isComputer) {
        bool ValidMove = false;
        while (!ValidMove) {
            int* validMoves = board.printValidBoard(roll, currentPlayer);
            // get user input
            logger.print(LogLevel::Player, "What room would you like to move to?");
            int move;
            std::cin >> move;
            // The user sees rooms 1-9, but the validMoves array is indexed 0-8.
            if (move > 0 && move <= 9 && validMoves[move - 1] == 1) {
                ValidMove = true;
                board.movePlayer(currentPlayer, move - 1);
            } else {
                logger.print(LogLevel::Player, "That is not a valid move, please type in a room highlighed in Green.");
                logger.print(LogLevel::Player, name + " rolled a " + std::to_string(roll));
            }
            
            delete[] validMoves; // Clean up the dynamically allocated array
        }
    } else {
        //COM logic for moving
    }
    board.printBoard();
}

Suggestion Player::makeSuggestion() {
    Suggestion suggestion;
    if (!isComputer) {
        //player input for making a suggestion
    } else {
        //COM logic for making a suggestion
    }
    return suggestion;
}
