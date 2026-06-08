#pragma once
#include "Board.h"
#include "Constants.h"
#include "Log.h"
#include "Player.h"
#include "Types.h"

class Game {
private:
  // Game Constants
  int numPlayers;
  int numCOMPlayers;
  const int MAX_PLAYERS = 6;
  int secretSuspect;
  int secretWeapon;
  int secretRoom;
  Player *players;
  // Game State Variables
  int round;
  int turn;
  int currentPlayer;
  bool gameOver;
  Board board;

  void initGame();
  void selectSecretCards();
  void dealCards();

public:
  Logger logger;
  // Constructor declaration
  Game(int players, int comPlayers);

  // Method declaration
  void loop();

  Disprove checkSuggestion(Suggestion suggestion, int player);
  void showCard(Disprove disprove, Suggestion suggestion);

  void finalAccusation();

  bool checkAccusation(Suggestion accusation);

  void updatePlayerData(Suggestion suggestion, int helperIndex, int askerIndex);
  void addAntiToAllPlayers(Suggestion suggestion, int player);
};