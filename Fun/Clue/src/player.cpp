// player.cpp
#include "Player.h"
#include "Board.h"
#include "Game.h"
#include "Log.h"

// Constructor Implementation
Player::Player()
    : name(""), isComputer(false), position(5), numCards(0), active(true) {
  cards = nullptr;
  susPeople = nullptr;
  susWeapons = nullptr;
  susRooms = nullptr;
}
Player::Player(int playerIndex, bool computer, LogLevel logLevel,
               int numPlayers)
    : playerIndex(playerIndex), isComputer(computer), position(5), numCards(0),
      logger(logLevel), active(true), numPlayers(numPlayers) {
  name = (isComputer ? "COM " : "Player ") + std::to_string(playerIndex + 1);
  cards = new LinkedList<int>();
  susPeople = loadList(ClueData::SUSPECTS.size());
  susWeapons = loadList(ClueData::WEAPONS.size());
  susRooms = loadList(ClueData::ROOMS.size());
  playerData = loadPlayerData();
}

// Helper Method Implementations
int Player::cardToNum(std::string card, int type) {
  if (type == 0) {
    for (size_t i = 0; i < ClueData::SUSPECTS.size(); i++) {
      if (ClueData::SUSPECTS[i] == card)
        return i;
    }
  } else if (type == 1) {
    for (size_t i = 0; i < ClueData::WEAPONS.size(); i++) {
      if (ClueData::WEAPONS[i] == card)
        return i;
    }
  } else if (type == 2) {
    for (size_t i = 0; i < ClueData::ROOMS.size(); i++) {
      if (ClueData::ROOMS[i] == card)
        return i;
    }
  }
  return -1;
}

std::string Player::numToCard(int num, int type) {
  if (type == 0) {
    if (num >= 0 && num < ClueData::SUSPECTS.size())
      return ClueData::SUSPECTS[num];
  } else if (type == 1) {
    if (num >= 0 && num < ClueData::WEAPONS.size())
      return ClueData::WEAPONS[num];
  } else if (type == 2) {
    if (num >= 0 && num < ClueData::ROOMS.size())
      return ClueData::ROOMS[num];
  }
  return "";
}

LinkedList<int> *Player::loadList(int len) {
  LinkedList<int> *list = new LinkedList<int>();
  for (int i = 0; i < len; i++) {
    list->add(i);
  }
  return list;
}

LinkedList<PlayerData *> *Player::loadPlayerData() {
  LinkedList<PlayerData *> *list = new LinkedList<PlayerData *>();
  for (int i = 0; i < numPlayers; i++) {
    list->add(new PlayerData(i));
  }
  return list;
}

// Gameplay Method Implementations
Suggestion Player::takeTurn(Game &game, Board &board, int currentPlayer) {
  if (!active) {
    logger.print(LogLevel::Player,
                 name + " is no longer active and cannot take a turn.");
    return Suggestion{-1, -1, -1}; // Return an invalid suggestion
  }
  LogLevel playerLevel = isComputer ? LogLevel::COM : LogLevel::Player;

  logger.print(playerLevel, "Please pass the device to " + name + ".");
  logger.print(playerLevel, name + " is taking their turn.");
  if (!isComputer) {
    logger.holdScreen();
    logger.clearScreen();
  }

  int roll = rand() % 3 + 1;
  logger.print(playerLevel, name + " rolled a " + std::to_string(roll));
  // Get new room position

  movePlayer(roll, board, currentPlayer);

  Suggestion suggestion;

  suggestion = makeSuggestion();

  return suggestion;
}

void Player::learnCard(int card) {
  // Remove the card from the player's list of possible cards and do nothing if
  // the card is already shown to the player
  if (card % 10 == 0) {
    // check if the card is already shown to the player
    if (susPeople->contains(card)) {
      susPeople->remove(card);
    }
  } else if (card % 10 == 1) {
    if (susWeapons->contains(card % 10)) {
      susWeapons->remove(card % 10);
    }
  } else if (card % 10 == 2) {
    if (susRooms->contains(card % 20)) {
      susRooms->remove(card % 20);
    }
  }
}

void Player::addCard(int card) {
  cards->add(card);
  playerData->get(playerIndex)->cards->add(card);
  learnCard(card);
  numCards++;
}

std::string Player::getName() { return name; }
int Player::getNumberOfCards() { return numCards; }

void Player::movePlayer(int roll, Board &board, int currentPlayer) {
  if (!isComputer) {
    bool ValidMove = false;
    while (!ValidMove) {
      int *validMoves = board.printValidBoard(roll, currentPlayer);
      // get user input
      logger.print(LogLevel::Player, "What room would you like to move to?");
      int move;
      std::cin >> move;
      // The user sees rooms 1-9, but the validMoves array is indexed 0-8.
      if (move > 0 && move <= 9 && validMoves[move - 1] == 1) {
        ValidMove = true;
        board.movePlayer(currentPlayer, move - 1);
      } else {
        logger.print(LogLevel::Player, "That is not a valid move, please type "
                                       "in a room highlighed in Green.");
        logger.print(LogLevel::Player,
                     name + " rolled a " + std::to_string(roll));
      }

      delete[] validMoves; // Clean up the dynamically allocated array
    }
  } else {
    // COM logic for moving
  }
  board.printBoard();
}

Suggestion Player::makeSuggestion() {
  Suggestion suggestion;
  suggestion.room =
      position; // The room is always the current position of the player
  if (!isComputer) {
    // player input for making a suggestion
    while (true) {
      logger.print(LogLevel::Player, "Enter a suspect to suggest:");
      std::string suspect;
      std::cin >> suspect;
      suggestion.suspect = cardToNum(suspect, 0);
      if (suggestion.suspect != -1)
        break;
      logger.print(LogLevel::Player, "Invalid suspect. Please try again.");
    }
    while (true) {
      logger.print(LogLevel::Player, "Enter a weapon to suggest:");
      std::string weapon;
      std::cin >> weapon;
      suggestion.weapon = cardToNum(weapon, 1);
      if (suggestion.weapon != -1)
        break;
      logger.print(LogLevel::Player, "Invalid weapon. Please try again.");
    }
  } else {
    // COM logic for making a suggestion
  }
  return suggestion;
}

LinkedList<int> *Player::checkHandForCard(Suggestion suggestion) {
  LinkedList<int> *tempCards = new LinkedList<int>();
  for (int i = 0; i < numCards; i++) {
    int card = cards->pop(); // Take Card off the top
    cards->add(card);        // Put it back on the bottom
    if (card == suggestion.suspect || card == suggestion.weapon ||
        card == suggestion.room) {
      tempCards->add(card); // Store matching cards in a temporary list
    }
  }
  if (tempCards->getSize() > 0) {
    return tempCards; // Return the list of matching cards
  }
  return nullptr; // No matching card found
}

bool Player::isComputerPlayer() { return isComputer; }

void Player::learnFromSuggestions() {
  int numOfSuggestions;
  int susCardLocation;
  int weaponCardLocation;
  int roomCardLocation;
  for (int i = 0; i < numPlayers; i++) {
    numOfSuggestions = playerData->get(i)->possibleSuspects->getSize();
    if (numOfSuggestions == 0) {
      continue;
    }
    for (int j = 0; j < numOfSuggestions; j++) {
      Suggestion suggestion = playerData->get(i)->possibleSuspects->get(j);
      susCardLocation = checkWhereCardIs(suggestion.suspect);
      weaponCardLocation = checkWhereCardIs(suggestion.weapon);
      roomCardLocation = checkWhereCardIs(suggestion.room);
      int unknownCount = 0;
      if (susCardLocation < 0) {
        unknownCount++;
      }
      if (weaponCardLocation < 0) {
        unknownCount++;
      }
      if (roomCardLocation < 0) {
        unknownCount++;
      }
      if (unknownCount == 0) {
        // remove this suggestion from the list
        playerData->get(i)->possibleSuspects->remove(suggestion);
        j--;
        numOfSuggestions--; // fix loop indexing after suggestion removal
        continue;
      }
      if (unknownCount > 1) {
        continue;
      }

      if (susCardLocation < 0) {
        // We know this card must be in this players hand since we know the
        // locations of the other two
        playerData->get(i)->cards->add(suggestion.suspect);
        // We also know that every other player does not have this card
        for (int k = 0; k < numPlayers; k++) {
          if (k == i) {
            continue;
          }
          playerData->get(k)->antiCards->add(suggestion.suspect);
        }
      }
    }
  }
}

int Player::checkWhereCardIs(int card) {
  for (int i = 0; i < numPlayers; i++) {
    if (playerData->get(i)->cards->contains(card)) {
      return i;
    }
  }
  return -1;
}