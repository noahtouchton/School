#include "../include/Game.h"
#include "../include/Board.h"
#include <chrono>
#include <format>
#include <iostream>
#include <thread>
#include <limits>

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

void Game::showPlayersCards() {
  for (int i = 0; i < (numPlayers + numCOMPlayers); i++) {
    if (!players[i].isComputerPlayer()) {
      logger.print(LogLevel::Player,
                   "Please pass the device to: " + players[i].getName());
      logger.holdScreen();
      logger.clearScreen();
      logger.print(LogLevel::Player, "Here are your cards: ");
      players[i].showHand();
      logger.print(LogLevel::Player,
                   "Please pass the device to the next player.");
      logger.holdScreen();
      logger.clearScreen();
    } else {
      logger.print(LogLevel::COM, "Player " + players[i].getName() +
                                      " is checking their cards.");
      std::this_thread::sleep_for(std::chrono::seconds(2));
      logger.clearScreen();
    }
  }
}

void Game::loop() {
  // Main game loop that continues until the game is over
  logger.print(LogLevel::Game, "Welcome to Clue!");
  board.printBoard();
  showPlayersCards();

  while (!gameOver) {
    logger.print(LogLevel::Game, "Round " + std::to_string(round) + ": " +
                                     players[currentPlayer].getName() +
                                     "'s turn");
    // Moves the player to a new room and gets their guess.
    Suggestion suggestion =
        players[currentPlayer].takeTurn(*this, board, currentPlayer);
    if (!players[currentPlayer].isComputerPlayer()) {
      logger.print(LogLevel::Game, "Please show the device to everyone.");
      logger.holdScreen();
      logger.clearScreen();
    } else {
      logger.clearScreen();
    }
    logger.print(LogLevel::Game, players[currentPlayer].getName() + " suggested: " +
                 players[currentPlayer].numToCard(suggestion.suspect, 0) + " with the " +
                 players[currentPlayer].numToCard(suggestion.weapon - 10, 1) + " in the " +
                 players[currentPlayer].numToCard(suggestion.room - 20, 2));
    logger.print(LogLevel::Game, "");
    logger.holdScreen();
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
    if (currentPlayer == 0) {
      round++;
    }
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
    if (disprove.possibleCards != nullptr) {
      delete disprove.possibleCards;
    }
    // Update player databases so everyone learns that no one has these cards
    updatePlayerData(suggestion, -1, currentPlayer);
    return;
  }

  int choice = 1;
  int cardToShow = -1;

  if (players[disprove.player].isComputerPlayer()) {
    // COM disproves silently
    choice = 1;
    cardToShow = disprove.possibleCards->get(0);
    logger.print(LogLevel::Game, players[disprove.player].getName() + " disproved the suggestion.");
    logger.holdScreen();
    // If the asker is a computer player, clear screen after pause
    if (players[currentPlayer].isComputerPlayer()) {
      logger.clearScreen();
    }
  } else {
    // Human disproves
    LogLevel disproverPlayerLevel = LogLevel::Player;
    logger.print(disproverPlayerLevel, "Please pass the device to " + players[disprove.player].getName() + " to reveal a card that disproves the suggestion.");
    logger.holdScreen();
    logger.clearScreen();
    logger.print(disproverPlayerLevel, "Choose a card to show:");
    int numPossible = disprove.possibleCards->getSize();
    for (int i = 0; i < numPossible; i++) {
      int card = disprove.possibleCards->get(i);
      logger.print(disproverPlayerLevel, "Card " + std::to_string(i + 1) + ": " + players[disprove.player].numToCard(card % 10, card / 10));
    }
    while (true) {
      logger.print(disproverPlayerLevel, "Enter the number of the card you want to show (1-" + std::to_string(numPossible) + "):");
      std::cin >> choice;
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      if (choice >= 1 && choice <= numPossible) {
        break;
      }
      logger.print(disproverPlayerLevel, "Invalid choice. Please enter a number between 1 and " + std::to_string(numPossible));
    }
    cardToShow = disprove.possibleCards->get(choice - 1);
    logger.clearScreen();
    // If the asker is a computer player, print that the human disproved it and pause
    if (players[currentPlayer].isComputerPlayer()) {
      logger.print(LogLevel::Game, players[disprove.player].getName() + " disproved the suggestion.");
      logger.holdScreen();
      logger.clearScreen();
    }
  }

  // Asker learns the card
  players[currentPlayer].learnCard(cardToShow);
  if (!players[currentPlayer].playerData->get(disprove.player)->cards->contains(cardToShow)) {
    players[currentPlayer].playerData->get(disprove.player)->cards->add(cardToShow);
  }
  
  // Track that this player has shown this card
  players[disprove.player].timesShown[cardToShow]++;

  // If the asker is human, reveal the card to them
  if (!players[currentPlayer].isComputerPlayer()) {
    LogLevel currentPlayerLevel = LogLevel::Player;
    logger.print(currentPlayerLevel, "Please pass the device to " + players[currentPlayer].getName() + " to reveal the card that disproves the suggestion.");
    logger.holdScreen();
    logger.clearScreen();
    logger.print(currentPlayerLevel, players[disprove.player].getName() + " showed you the card: " + players[disprove.player].numToCard(cardToShow % 10, cardToShow / 10));
    logger.holdScreen();
    logger.clearScreen();
  }

  // Update databases for everyone
  updatePlayerData(suggestion, disprove.player, currentPlayer);

  // Clean up possibleCards memory (prevents memory leak and copy constructor segfault)
  delete disprove.possibleCards;
}

void Game::finalAccusation() {
  int ready;
  LogLevel currentPlayerLevel;
  Suggestion accusation;
  if (players[currentPlayer].isComputerPlayer()) {
    currentPlayerLevel = LogLevel::COM;
    // COM logic determining if the COM player is ready to make an accusation
    ready = 0;
    if (players[currentPlayer].checkAccusationReady(accusation)) {
      ready = 1;
    }
  } else {
    currentPlayerLevel = LogLevel::Player;
    logger.print(currentPlayerLevel,
                 "Are you ready to make an accusation? (1 for yes, 0 for no, 9 "
                 "to toggle COM Debug Mode)");
    while (true) {
      std::cin >> ready;
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      if (ready == 9) {
        comDebugMode = !comDebugMode;
        std::cout << "\033[1;35m[SYSTEM] COM Debug Mode is now "
                  << (comDebugMode ? "ENABLED" : "DISABLED") << "\033[0m"
                  << std::endl;
        logger.print(currentPlayerLevel,
                     "Are you ready to make an accusation? (1 for yes, 0 for "
                     "no, 9 to toggle COM Debug Mode)");
        continue;
      }
      if (ready == 1 || ready == 0) {
        break;
      }
      logger.print(currentPlayerLevel,
                   "Invalid input. Please enter 1 for yes, 0 for no, or 9 to "
                   "toggle COM Debug Mode.");
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
        std::getline(std::cin >> std::ws, suspect);
        accusation.suspect = players[currentPlayer].cardToNum(suspect, 0);
        if (accusation.suspect != -1)
          break;
        logger.print(currentPlayerLevel, "Invalid suspect. Please try again.");
      }
      while (true) {
        logger.print(currentPlayerLevel, "Enter a weapon for your accusation:");
        std::string weapon;
        std::getline(std::cin >> std::ws, weapon);
        int val = players[currentPlayer].cardToNum(weapon, 1);
        if (val != -1) {
          accusation.weapon = val + 10; // Convert to absolute card ID
          break;
        }
        logger.print(currentPlayerLevel, "Invalid weapon. Please try again.");
      }
      while (true) {
        logger.print(currentPlayerLevel, "Enter a room for your accusation:");
        std::string room;
        std::getline(std::cin >> std::ws, room);
        int val = players[currentPlayer].cardToNum(room, 2);
        if (val != -1) {
          accusation.room = val + 20; // Convert to absolute card ID
          break;
        }
        logger.print(currentPlayerLevel, "Invalid room. Please try again.");
      }
    }
  }
  if (checkAccusation(accusation)) {
    logger.print(currentPlayerLevel,
                 "Congratulations! Your accusation is correct. You win!");
    logger.print(currentPlayerLevel, "\nNumber of rounds: " + std::to_string(round) + "\n");
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
          accusation.weapon == secretWeapon + 10 &&
          accusation.room == secretRoom + 20);
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
