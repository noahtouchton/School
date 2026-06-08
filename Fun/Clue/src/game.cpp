#include "../include/Game.h"
#include "../include/Board.h"
#include <format>
#include <iostream>

// Things that must be done in main prior to creating a Game object:
// 1. Ask the user how many players (1-6)
// 2. Ask the user how many COM (total players must be between 1 and 6,
// inclusive)

Game::Game(int players, int comPlayers)
    : numPlayers(players), numCOMPlayers(comPlayers), currentPlayer(0),
      gameOver(false), round(1), board(players + comPlayers, LogLevel::Game),
      logger(LogLevel::COM) {
  // Run the logic to randomly select the secret suspect, weapon, and room
  selectSecretCards();
  initGame();
  dealCards();
}

void Game::initGame() {
  // Logic to initialize the game state, such as shuffling cards and setting up
  // players assumes number of players is less than 6 as this will be checked in
  // main before creating the Game object
  players = new Player[numPlayers + numCOMPlayers];
  for (int i = 0; i < numPlayers; i++) {
    players[i] = Player(i, false, LogLevel::Player, numPlayers + numCOMPlayers);
  }
  for (int i = numPlayers; i < numPlayers + numCOMPlayers; i++) {
    players[i] = Player(i, true, LogLevel::COM, numPlayers + numCOMPlayers);
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

  LinkedList<int> *deck = new LinkedList<int>();
  for (int i = 0; i < ClueData::SUSPECTS.size(); i++) {
    if (i != secretSuspect) {
      deck->add(i);
    }
  }
  // weapons start at 10
  for (int i = 0; i < ClueData::WEAPONS.size(); i++) {
    if (i != secretWeapon) {
      deck->add(i + 10);
    }
  }
  // rooms start at 20
  for (int i = 0; i < ClueData::ROOMS.size(); i++) {
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
    logger.print(LogLevel::Game, "Round " + std::to_string(round) + ": " +
                                     players[currentPlayer].getName() +
                                     "'s turn");
    // Moves the player to a new room and gets their guess.
    Suggestion suggestion =
        players[currentPlayer].takeTurn(*this, board, currentPlayer);
    logger.print(LogLevel::Game, "Please show the device to everyone.");
    logger.holdScreen();
    logger.clearScreen();
    // Sees which player can disprove the suggestion, starting with the next
    // player and going clockwise
    Disprove disprove = checkSuggestion(
        suggestion, (currentPlayer + 1) % (numPlayers + numCOMPlayers));

    // Have the desire player choose which card to show
    showCard(disprove, suggestion);

    // Ask the player if they want to make an accusation, and if so, check if
    // it's correct. If it's correct, end the game and declare them the winner.
    // If it's incorrect, remove them from the game and continue with the
    // remaining players.
    finalAccusation();

    // Move to the next player
    currentPlayer = (currentPlayer + 1) % (numPlayers + numCOMPlayers);
  }
}

Disprove Game::checkSuggestion(Suggestion suggestion, int player) {
  // making a recursive function
  Disprove disprove = Disprove();
  if (player == currentPlayer) {
    return disprove; // Can't check the suggestion against the current player
  }
  logger.print(LogLevel::Game,
               players[player].getName() + " is checking the suggestion...");
  LinkedList<int> *matchingCards = players[player].checkHandForCard(suggestion);
  if (matchingCards != nullptr) {
    disprove.possibleCards = matchingCards;
    disprove.player = player;
    logger.print(LogLevel::Game,
                 players[player].getName() + " can disprove the suggestion.");
    return disprove;
  } else {
    int nextPlayer = (player + 1) % (numPlayers + numCOMPlayers);
    logger.print(LogLevel::Game, players[player].getName() +
                                     " cannot disprove the suggestion.");
    return checkSuggestion(suggestion, nextPlayer);
  }
}

void Game::showCard(Disprove disprove, Suggestion suggestion) {
  if (disprove.player == -1 || disprove.possibleCards == nullptr) {
    logger.print(LogLevel::Game, "No one was able to disprove the suggestion.");
    logger.holdScreen();
    logger.clearScreen();
    return;
  }
  LogLevel disproverPlayerLevel;
  if (players[disprove.player].isComputerPlayer()) {
    disproverPlayerLevel = LogLevel::COM;
  } else {
    disproverPlayerLevel = LogLevel::Player;
  }

  LogLevel currentPlayerLevel;
  if (players[currentPlayer].isComputerPlayer()) {
    currentPlayerLevel = LogLevel::COM;
  } else {
    currentPlayerLevel = LogLevel::Player;
  }

  logger.print(disproverPlayerLevel,
               "Please show the device to " +
                   players[disprove.player].getName() +
                   " to reveal a card that disproves the suggestion.");
  logger.holdScreen();
  logger.clearScreen();
  logger.print(disproverPlayerLevel, "Choose a card to show:");
  int choice = 1;
  int numPossible = disprove.possibleCards->getSize();
  for (int i = 0; i < numPossible; i++) {
    int card = disprove.possibleCards->pop();
    disprove.possibleCards->add(card);
    logger.print(disproverPlayerLevel,
                 "Card " + std::to_string(i + 1) + ": " +
                     players[disprove.player].numToCard(card % 10, card / 10));
  }

  if (players[disprove.player].isComputerPlayer()) {
    // COM logic for choosing which card to show
    choice = 1;
  } else {
    while (true) {
      logger.print(disproverPlayerLevel,
                   "Enter the number of the card you want to show (1-" +
                       std::to_string(numPossible) + "):");
      std::cin >> choice;
      if (choice >= 1 && choice <= numPossible) {
        break;
      }
      logger.print(disproverPlayerLevel,
                   "Invalid choice. Please enter a number between 1 and " +
                       std::to_string(numPossible));
    }
  }

  int cardToShow = -1;
  for (int j = 0; j < choice; j++) {
    cardToShow = disprove.possibleCards->pop();
    disprove.possibleCards->add(cardToShow);
  }
  // learn the card that was shown
  players[currentPlayer].learnCard(cardToShow);

  logger.print(currentPlayerLevel,
               "Please show the device to " + players[currentPlayer].getName() +
                   " to reveal the card that disproves the suggestion.");
  logger.holdScreen();
  logger.clearScreen();
  logger.print(
      currentPlayerLevel,
      players[disprove.player].getName() + " showed you the card: " +
          players[disprove.player].numToCard(cardToShow % 10, cardToShow / 10));
  logger.holdScreen();
  logger.clearScreen();

  // have everyone learn what the can from the current sugestion
  updatePlayerData(suggestion, disprove.player, currentPlayer);
}

void Game::finalAccusation() {
  int ready;
  LogLevel currentPlayerLevel;
  Suggestion accusation;
  if (players[currentPlayer].isComputerPlayer()) {
    currentPlayerLevel = LogLevel::COM;
    // COM logic determining if the COM player is ready to make an accusation
    ready = 0;
  } else {
    currentPlayerLevel = LogLevel::Player;
    logger.print(currentPlayerLevel,
                 "Are you ready to make an accusation? (1 for yes, 0 for no)");
    while (true) {
      std::cin >> ready;
      if (ready == 1 || ready == 0) {
        break;
      }
      logger.print(currentPlayerLevel,
                   "Invalid input. Please enter 1 for yes or 0 for no.");
    }
  }

  if (ready == 0) {
    return; // player is not ready
  } else {
    // Player is ready, proceed with the accusation
    if (!players[currentPlayer].isComputerPlayer()) {
      // player input for making an accusation
      while (true) {
        logger.print(currentPlayerLevel,
                     "Enter a suspect for your accusation:");
        std::string suspect;
        std::cin >> suspect;
        accusation.suspect = players[currentPlayer].cardToNum(suspect, 0);
        if (accusation.suspect != -1)
          break;
        logger.print(currentPlayerLevel, "Invalid suspect. Please try again.");
      }
      while (true) {
        logger.print(currentPlayerLevel, "Enter a weapon for your accusation:");
        std::string weapon;
        std::cin >> weapon;
        accusation.weapon = players[currentPlayer].cardToNum(weapon, 1);
        if (accusation.weapon != -1)
          break;
        logger.print(currentPlayerLevel, "Invalid weapon. Please try again.");
      }
      while (true) {
        logger.print(currentPlayerLevel, "Enter a room for your accusation:");
        std::string room;
        std::cin >> room;
        accusation.room = players[currentPlayer].cardToNum(room, 2);
        if (accusation.room != -1)
          break;
        logger.print(currentPlayerLevel, "Invalid room. Please try again.");
      }
    } else {
      // COM logic for choosing a suspect to accuse
      accusation.suspect = 0;
      accusation.weapon = 0;
      accusation.room = 0;
    }
  }
  if (checkAccusation(accusation)) {
    logger.print(currentPlayerLevel,
                 "Congratulations! Your accusation is correct. You win!");
    gameOver = true;
  } else {
    logger.print(
        currentPlayerLevel,
        "Sorry, your accusation is incorrect. You are out of the game.");
    // Logic to remove the player from the game and continue with the remaining
    // players
    players[currentPlayer].setInactive();
    // Check if all other players are inactive, if so, end the game and declare
    // the remaining active player the winner
    int activePlayers = 0;
    int lastActivePlayer = -1;
    for (int i = 0; i < numPlayers + numCOMPlayers; i++) {
      if (players[i].isActive()) {
        activePlayers++;
        lastActivePlayer = i;
      }
    }
    if (activePlayers == 1) {
      logger.print(currentPlayerLevel, "Congratulations! " +
                                           players[lastActivePlayer].getName() +
                                           " is the winner!");
      gameOver = true;
    }
  }
}

bool Game::checkAccusation(Suggestion accusation) {
  return (accusation.suspect == secretSuspect &&
          accusation.weapon == secretWeapon && accusation.room == secretRoom);
}

void Game::updatePlayerData(Suggestion suggestion, int helperIndex,
                            int askerIndex) {
  int totalPlayers = numPlayers + numCOMPlayers;
  for (int i = 1; i < totalPlayers; i++) {
    int relIndex = (askerIndex + i) % totalPlayers;
    if (relIndex == helperIndex) {
      // add the suggestion to possible suspects
      for (int p = 0; p < totalPlayers; p++)
        players[p].playerData->get(relIndex)->possibleSuspects->add(suggestion);
      // Call a function that checks if this possible suspects function helped
      // learn a card

      break; // Stop at the helper who disproved
    }

    // Mark that relIndex doesn't have any of the cards in the suggestion
    // Update all players' databases
    for (int p = 0; p < totalPlayers; p++) {
      players[p].playerData->get(relIndex)->antiCards->add(suggestion.suspect);
      players[p].playerData->get(relIndex)->antiCards->add(suggestion.weapon);
      players[p].playerData->get(relIndex)->antiCards->add(suggestion.room);
    }
  }
}
