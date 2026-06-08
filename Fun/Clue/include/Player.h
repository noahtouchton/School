#pragma once
#include "Constants.h"
#include "LinkedList.h"
#include "Log.h"
#include "Types.h"
#include <iostream>
#include <string>

class Game;
class Board;

class Player {
private:
  int position;
  bool isComputer;
  int numCards;
  bool active;
  int playerIndex;
  int numPlayers;
  std::string name;
  LinkedList<int> *cards;

  LinkedList<int> *susPeople;
  LinkedList<int> *susWeapons;
  LinkedList<int> *susRooms;

  LinkedList<int> *loadList(int len);

public:
  Logger logger;
  LinkedList<PlayerData *> *playerData;
  LinkedList<PlayerData *> *loadPlayerData();
  // Constructor declaration
  Player();
  Player(int playerIndex, bool computer, LogLevel logLevel, int numPlayers);

  // Method declaration
  Suggestion takeTurn(Game &game, Board &board, int currentPlayer);
  void learnCard(int card);
  void addCard(int card);

  int cardToNum(std::string card, int type);
  std::string numToCard(int num, int type);

  void movePlayer(int roll, Board &board, int currentPlayer);
  Suggestion makeSuggestion();

  std::string getName();

  int getNumberOfCards();

  LinkedList<int> *checkHandForCard(Suggestion suggestion);

  bool isComputerPlayer();

  bool isActive() { return active; }
  void setInactive() { active = false; }

  void learnFromSuggestions();
  int checkWhereCardIs(int card);
};