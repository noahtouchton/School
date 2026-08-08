// player.cpp
#include "Player.h"
#include "Board.h"
#include "Game.h"
#include "Log.h"
#include <algorithm>
#include <future>
#include <limits>
#include <random>
#include <thread>
#include <utility>

bool comDebugMode = true;

// Constructor Implementation
Player::Player()
    : name(""), isComputer(false), position(5), numCards(0), active(true) {
  cards = nullptr;
  susPeople = nullptr;
  susWeapons = nullptr;
  susRooms = nullptr;
  timesShown = std::vector<int>(29, 0);
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
  timesShown = std::vector<int>(29, 0);
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

  if (!isComputer) {
    logger.print(LogLevel::Player, "Please pass the device to " + name + ".");
    logger.print(LogLevel::Player, name + " is taking their turn.");
    logger.holdScreen();
    logger.clearScreen();
  } else {
    logger.print(LogLevel::Game, "\n" + name + " is taking their turn...");
  }

  int roll = rand() % 3 + 1;
  if (!isComputer) {
    logger.print(LogLevel::Player, name + " rolled a " + std::to_string(roll));
  } else {
    if (comDebugMode) {
      logger.print(LogLevel::Game, name + " rolled a " + std::to_string(roll));
    }
  }

  Suggestion suggestion = movePlayer(roll, board, currentPlayer);
  return suggestion;
}

void Player::learnCard(int card) {
  // Remove the card from the player's list of possible cards and do nothing if
  // the card is already shown to the player
  if (card >= 0 && card <= 5) {
    if (susPeople->contains(card)) {
      susPeople->remove(card);
    }
  } else if (card >= 10 && card <= 15) {
    int val = card - 10;
    if (susWeapons->contains(val)) {
      susWeapons->remove(val);
    }
  } else if (card >= 20 && card <= 28) {
    int val = card - 20;
    if (susRooms->contains(val)) {
      susRooms->remove(val);
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

Suggestion Player::movePlayer(int roll, Board &board, int currentPlayer) {
  Suggestion suggestion;
  std::vector<bool> possessedRooms(9, false);
  for (int i = 0; i < 9; i++) {
    if (cards->contains(i + 20)) {
      possessedRooms[i] = true;
    }
  }

  if (!isComputer) {
    bool ValidMove = false;
    while (!ValidMove) {
      int *validMoves =
          board.printValidBoard(roll, currentPlayer, true, possessedRooms);
      // get user input
      logger.print(LogLevel::Player, "What room would you like to move to?");
      int move;
      std::cin >> move;
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      // The user sees rooms 1-9, but the validMoves array is indexed 0-8.
      if (move > 0 && move <= 9 && validMoves[move - 1] == 1) {
        ValidMove = true;
        board.movePlayer(currentPlayer, move - 1);
        position = move - 1; // Update position
        logger.print(LogLevel::Game,
                     name + " moved to " + numToCard(position, 2));
      } else {
        logger.print(LogLevel::Player, "That is not a valid move, please type "
                                       "in a room highlighed in Green.");
        logger.print(LogLevel::Player,
                     name + " rolled a " + std::to_string(roll));
      }
      delete[] validMoves; // Clean up the dynamically allocated array
    }
    suggestion = makeSuggestion();
    board.printBoard(possessedRooms);
  } else {
    // COM logic for moving
    std::vector<std::vector<float>> cardFrequencies =
        cardMonteCarlo(ClueData::MONTE_CARLO_ITERATIONS);
    std::vector<int> possibleRooms;
    int *validMoves = board.printValidBoard(roll, currentPlayer, false);
    for (int i = 0; i < 9; i++) {
      if (validMoves[i] == 1) {
        possibleRooms.push_back(i);
      }
    }
    suggestion = getNextGuess(cardFrequencies, possibleRooms);
    board.movePlayer(currentPlayer,
                     suggestion.room - 20); // pass relative index
    position = suggestion.room - 20;        // Update position
    logger.print(LogLevel::Game, name + " moved to " + numToCard(position, 2));
    // return suggestion if a COM

    delete[] validMoves; // Clean up the dynamically allocated array
    board.printBoard();
  }
  return suggestion;
}

Suggestion Player::makeSuggestion() {
  Suggestion suggestion;
  suggestion.room = position + 20; // The room is always the current position of
                                   // the player (absolute card ID)

  // Print compact context to fit the screen
  std::cout << "\n========================================" << std::endl;
  std::cout << "Your Hand: ";
  for (int i = 0; i < cards->getSize(); i++) {
    int card = cards->get(i);
    std::cout << numToCard(card % 10, card / 10)
              << (i == cards->getSize() - 1 ? "" : ", ");
  }
  std::cout << std::endl;

  std::cout << "Suspects : ";
  for (size_t i = 0; i < ClueData::SUSPECTS.size(); i++) {
    int cardId = i;
    if (cards->contains(cardId)) {
      std::cout << "\033[1;32m" << ClueData::SUSPECTS[i] << "\033[0m";
    } else {
      std::cout << ClueData::SUSPECTS[i];
    }
    std::cout << (i == ClueData::SUSPECTS.size() - 1 ? "" : " | ");
  }
  std::cout << std::endl;

  std::cout << "Weapons  : ";
  for (size_t i = 0; i < ClueData::WEAPONS.size(); i++) {
    int cardId = i + 10;
    if (cards->contains(cardId)) {
      std::cout << "\033[1;32m" << ClueData::WEAPONS[i] << "\033[0m";
    } else {
      std::cout << ClueData::WEAPONS[i];
    }
    std::cout << (i == ClueData::WEAPONS.size() - 1 ? "" : " | ");
  }
  std::cout << std::endl;

  std::cout << "Room     : ";
  if (cards->contains(suggestion.room)) {
    std::cout << "\033[1;32m" << numToCard(position, 2) << "\033[0m"
              << std::endl;
  } else {
    std::cout << numToCard(position, 2) << std::endl;
  }
  std::cout << "========================================\n" << std::endl;

  // player input for making a suggestion
  while (true) {
    logger.print(LogLevel::Player, "Enter a suspect to suggest:");
    std::string suspect;
    std::getline(std::cin >> std::ws, suspect);
    suggestion.suspect = cardToNum(suspect, 0);
    if (suggestion.suspect != -1)
      break;
    logger.print(LogLevel::Player, "Invalid suspect. Please try again.");
  }
  while (true) {
    logger.print(LogLevel::Player, "Enter a weapon to suggest:");
    std::string weapon;
    std::getline(std::cin >> std::ws, weapon);
    suggestion.weapon =
        cardToNum(weapon, 1) + 10; // Convert to absolute card ID
    if (suggestion.weapon != -1 && suggestion.weapon >= 10 &&
        suggestion.weapon < 16)
      break;
    logger.print(LogLevel::Player, "Invalid weapon. Please try again.");
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

std::vector<std::vector<float>> Player::cardMonteCarlo(int numIterations) {
  // Place all the known cards
  std::vector<int> cardAssignments(29, -1);
  std::vector<int> playerCards(numPlayers, 0);
  std::vector<int> maxCards(numPlayers);
  std::vector<std::vector<bool>> antiCardsLookup(numPlayers,
                                                 std::vector<bool>(29, false));
  int baseCards = 18 / numPlayers;
  int remainder = 18 % numPlayers;
  for (int i = 0; i < numPlayers; i++) {
    maxCards[i] = baseCards + (i < remainder ? 1 : 0);
  }
  for (int i = 0; i < numPlayers; i++) {
    PlayerData *onePlayerData = playerData->get(i);
    for (int j = 0; j < onePlayerData->cards->getSize(); j++) {
      int card = onePlayerData->cards->get(j);
      cardAssignments[card] = i;
      playerCards[i]++;
    }
  }

  for (int i = 0; i < numPlayers; i++) {
    PlayerData *onePlayerData = playerData->get(i);
    for (int j = 0; j < onePlayerData->antiCards->getSize(); j++) {
      int card = onePlayerData->antiCards->get(j);
      antiCardsLookup[i][card] = true;
    }
  }

  // Create Suggestion Linked List for all other players
  LinkedList<playerSuggestion> *suggestions =
      new LinkedList<playerSuggestion>();
  for (int i = 0; i < numPlayers; i++) {
    if (i == playerIndex)
      continue;
    PlayerData *onePlayerData = playerData->get(i);
    for (int j = 0; j < onePlayerData->possibleSuspects->getSize(); j++) {
      playerSuggestion oneSuggestion;
      oneSuggestion.player = i;
      oneSuggestion.suggestion = onePlayerData->possibleSuspects->get(j);
      suggestions->add(oneSuggestion);
    }
  }
  std::vector<std::vector<int>> cardFrequencies(
      29, std::vector<int>(numPlayers + 1, 0));
  int successfulRuns = 0;

  std::cout << "[Monte Carlo] Simulating " << numIterations << " iterations..."
            << std::endl;
  int progressTicks = 4; // Change this to adjust print frequency (e.g. 4 for
                         // quarters, 100 for every percent)
  int progressStep = (progressTicks > 0) ? (numIterations / progressTicks) : 0;
  int numThreads = 8;
  int iterationsPerThread = numIterations / numThreads;

  std::vector<std::future<std::pair<int, std::vector<std::vector<int>>>>>
      futures;

  for (int t = 0; t < numThreads; t++) {
    futures.push_back(std::async(
        std::launch::async, [this, iterationsPerThread, t, numThreads,
                             progressStep, cardAssignments, playerCards,
                             maxCards, antiCardsLookup, suggestions]() {
          std::vector<std::vector<int>> localFrequencies(
              29, std::vector<int>(numPlayers + 1, 0));
          int localSuccess = 0;

          for (int i = 0; i < iterationsPerThread; i++) {
            if (t == 0 && progressStep > 0 &&
                i % (progressStep / numThreads) == 0 && i > 0) {
              std::cout << "  Progress: " << (i * 100 / iterationsPerThread)
                        << "%..." << std::endl;
            }
            std::vector<int> tempAssignments = cardAssignments;
            std::vector<int> tempPlayerCards = playerCards;

            if (generateCardDist(tempAssignments, tempPlayerCards, maxCards,
                                 antiCardsLookup, suggestions)) {
              localSuccess++;
              for (int card = 0; card < 29; card++) {
                if ((card >= 6 && card <= 9) || (card >= 16 && card <= 19))
                  continue;
                localFrequencies[card][tempAssignments[card]]++;
              }
            }
          }
          return std::make_pair(localSuccess, localFrequencies);
        }));
  }

  for (auto &f : futures) {
    auto result = f.get();
    successfulRuns += result.first;
    for (int card = 0; card < 29; card++) {
      for (int p = 0; p <= numPlayers; p++) {
        cardFrequencies[card][p] += result.second[card][p];
      }
    }
  }

  // Calculate and apply probabilities if we had any successful runs
  if (successfulRuns > 0) {
    logger.print(LogLevel::COM, "\n--- Monte Carlo Card Probabilities (" +
                                    std::to_string(successfulRuns) +
                                    " successful runs) ---");
    if (comDebugMode) {
      std::cout << "\033[1;35m\n--- [DEBUG] COM Player " << (playerIndex + 1)
                << " Monte Carlo Probabilities (" << successfulRuns
                << " successful runs) ---" << std::endl;
    }

    for (int card = 0; card < 29; card++) {
      if ((card >= 6 && card <= 9) || (card >= 16 && card <= 19))
        continue;

      std::string cardName;
      if (card >= 0 && card <= 5)
        cardName = numToCard(card, 0);
      else if (card >= 10 && card <= 15)
        cardName = numToCard(card - 10, 1);
      else if (card >= 20 && card <= 28)
        cardName = numToCard(card - 20, 2);

      double envProb =
          (double)cardFrequencies[card][numPlayers] / successfulRuns;

      std::string probStr =
          cardName + ": Envelope=" + std::to_string(envProb * 100.0) + "%";
      if (comDebugMode) {
        std::cout << "  " << cardName << ": Envelope=" << (envProb * 100.0)
                  << "%";
      }
      for (int p = 0; p < numPlayers; p++) {
        double pProb = (double)cardFrequencies[card][p] / successfulRuns;
        probStr += ", Player " + std::to_string(p + 1) + "=" +
                   std::to_string(pProb * 100.0) + "%";
        if (comDebugMode) {
          std::cout << ", Player " << (p + 1) << "=" << (pProb * 100.0) << "%";
        }

        // 1. If 100% probability player p has the card, learn it
        if (pProb >= 0.9999) {
          if (!playerData->get(p)->cards->contains(card)) {
            playerData->get(p)->cards->add(card);
            learnCard(card);
            logger.print(LogLevel::COM, "Deduction: Learnt " + cardName +
                                            " is held by Player " +
                                            std::to_string(p + 1));
          }
        }

        // 2. If 0% probability player p has the card, add to their anti-cards
        if (pProb <= 0.0001) {
          if (!playerData->get(p)->antiCards->contains(card)) {
            playerData->get(p)->antiCards->add(card);
          }
        }
      }
      if (comDebugMode) {
        std::cout << std::endl;
      }
      if (!comDebugMode) {
        logger.print(LogLevel::COM, probStr);
      }
    }
    if (comDebugMode) {
      std::cout
          << "---------------------------------------------------------\033[0m"
          << std::endl;
    }
  }

  std::vector<std::vector<float>> cardProbabilities(
      29, std::vector<float>(numPlayers + 1, 0.0f));
  if (successfulRuns > 0) {
    for (int card = 0; card < 29; card++) {
      if ((card >= 6 && card <= 9) || (card >= 16 && card <= 19))
        continue;
      for (int loc = 0; loc <= numPlayers; loc++) {
        cardProbabilities[card][loc] =
            (float)cardFrequencies[card][loc] / successfulRuns;
      }
    }
  }

  delete suggestions;
  return cardProbabilities;
}

bool Player::generateCardDist(
    std::vector<int> &cardAssignments, std::vector<int> &playerCards,
    const std::vector<int> &maxCards,
    const std::vector<std::vector<bool>> &antiCardsLookup,
    LinkedList<playerSuggestion> *suggestions) {
  suggestions->shuffle();
  return backtrackSolve(0, cardAssignments, playerCards, maxCards,
                        antiCardsLookup, suggestions);
}

bool backtrackRemaining(int cardId, std::vector<int> &cardAssignments,
                        std::vector<int> &playerCards,
                        const std::vector<int> &maxCards,
                        const std::vector<std::vector<bool>> &antiCardsLookup,
                        int numPlayers) {
  // Base case: all cards processed
  if (cardId >= 29) {
    // Check if envelope has exactly 1 suspect, 1 weapon, and 1 room
    int suspectsInEnvelope = 0, weaponsInEnvelope = 0, roomsInEnvelope = 0;
    for (int c = 0; c < 6; c++)
      if (cardAssignments[c] == numPlayers)
        suspectsInEnvelope++;
    for (int c = 10; c < 16; c++)
      if (cardAssignments[c] == numPlayers)
        weaponsInEnvelope++;
    for (int c = 20; c < 29; c++)
      if (cardAssignments[c] == numPlayers)
        roomsInEnvelope++;

    return (suspectsInEnvelope == 1 && weaponsInEnvelope == 1 &&
            roomsInEnvelope == 1);
  }

  // Skip unused card IDs
  if ((cardId >= 6 && cardId <= 9) || (cardId >= 16 && cardId <= 19)) {
    return backtrackRemaining(cardId + 1, cardAssignments, playerCards,
                              maxCards, antiCardsLookup, numPlayers);
  }

  // If already assigned
  if (cardAssignments[cardId] != -1) {
    return backtrackRemaining(cardId + 1, cardAssignments, playerCards,
                              maxCards, antiCardsLookup, numPlayers);
  }

  // Try assigning to the envelope or players in a random order
  std::vector<int> choices;

  // Envelope choice: represented by index numPlayers
  bool envelopeCategoryFilled = false;
  if (cardId >= 0 && cardId <= 5) {
    for (int c = 0; c < 6; c++)
      if (cardAssignments[c] == numPlayers)
        envelopeCategoryFilled = true;
  } else if (cardId >= 10 && cardId <= 15) {
    for (int c = 10; c < 16; c++)
      if (cardAssignments[c] == numPlayers)
        envelopeCategoryFilled = true;
  } else if (cardId >= 20 && cardId <= 28) {
    for (int c = 20; c < 29; c++)
      if (cardAssignments[c] == numPlayers)
        envelopeCategoryFilled = true;
  }

  if (!envelopeCategoryFilled) {
    choices.push_back(numPlayers);
  }

  // Player choices
  for (int p = 0; p < numPlayers; p++) {
    if (playerCards[p] < maxCards[p] && !antiCardsLookup[p][cardId]) {
      choices.push_back(p);
    }
  }

  // Shuffle choices to ensure randomness
  thread_local std::mt19937 rng(std::random_device{}());
  std::shuffle(choices.begin(), choices.end(), rng);

  for (int choice : choices) {
    cardAssignments[cardId] = choice;
    if (choice < numPlayers) {
      playerCards[choice]++;
    }

    if (backtrackRemaining(cardId + 1, cardAssignments, playerCards, maxCards,
                           antiCardsLookup, numPlayers)) {
      return true;
    }

    // Undo assignment
    cardAssignments[cardId] = -1;
    if (choice < numPlayers) {
      playerCards[choice]--;
    }
  }

  return false;
}

bool Player::backtrackSolve(
    int suggestionIndex, std::vector<int> &cardAssignments,
    std::vector<int> &playerCards, const std::vector<int> &maxCards,
    const std::vector<std::vector<bool>> &antiCardsLookup,
    LinkedList<playerSuggestion> *suggestions) {
  // Base Case
  if (suggestionIndex >= suggestions->getSize()) {
    // Satisfied all suggestions, now distribute the remaining unassigned cards
    return backtrackRemaining(0, cardAssignments, playerCards, maxCards,
                              antiCardsLookup, numPlayers);
  }

  playerSuggestion s = suggestions->get(suggestionIndex);

  // If the suggestion is already satisfied by a card already assigned to this
  // player, skip to next suggestion
  if (cardAssignments[s.suggestion.suspect] == s.player ||
      cardAssignments[s.suggestion.weapon] == s.player ||
      cardAssignments[s.suggestion.room] == s.player) {
    return backtrackSolve(suggestionIndex + 1, cardAssignments, playerCards,
                          maxCards, antiCardsLookup, suggestions);
  }

  // Check that this player is eligible to receive one of these cards
  if (playerCards[s.player] >= maxCards[s.player]) {
    return false;
  }

  std::vector<int> possibleCards;
  if (cardAssignments[s.suggestion.suspect] == -1 &&
      !antiCardsLookup[s.player][s.suggestion.suspect]) {
    possibleCards.push_back(s.suggestion.suspect);
  }
  if (cardAssignments[s.suggestion.weapon] == -1 &&
      !antiCardsLookup[s.player][s.suggestion.weapon]) {
    possibleCards.push_back(s.suggestion.weapon);
  }
  if (cardAssignments[s.suggestion.room] == -1 &&
      !antiCardsLookup[s.player][s.suggestion.room]) {
    possibleCards.push_back(s.suggestion.room);
  }

  if (possibleCards.empty()) {
    return false;
  }

  // Shuffle the candidate cards
  thread_local std::mt19937 rng(std::random_device{}());
  std::shuffle(possibleCards.begin(), possibleCards.end(), rng);

  // Try Each Valid Card, recurse, and backtrack on failure
  for (size_t i = 0; i < possibleCards.size(); i++) {
    int card = possibleCards[i];

    // Assign card
    cardAssignments[card] = s.player;
    playerCards[s.player]++;

    // Recurse
    if (backtrackSolve(suggestionIndex + 1, cardAssignments, playerCards,
                       maxCards, antiCardsLookup, suggestions)) {
      return true;
    }

    // Backtrack (revert state)
    cardAssignments[card] = -1;
    playerCards[s.player]--;
  }

  return false;
}

Suggestion
Player::getNextGuess(std::vector<std::vector<float>> &cardFrequencies,
                     std::vector<int> &possibleRooms) {
  std::vector<int> probSus;
  std::vector<int> probWeapon;

  if (comDebugMode) {
    std::cout << "\033[1;36m\n--- [DEBUG] COM Player " << (playerIndex + 1)
              << " Guess Optimization ---" << std::endl;
    std::cout << "  Reachable rooms: ";
    for (int r : possibleRooms) {
      std::cout << numToCard(r, 2) << " ";
    }
    std::cout << "\n  Evaluating candidate suggestions (Current Board Entropy: "
              << calculateBoardEntropy(cardFrequencies) << "):" << std::endl;
  }

  int maxIndex = 0;
  int secondMax = 1;
  for (int i = 0; i < 6; i++) {
    if (cardFrequencies[i][numPlayers] >
        cardFrequencies[maxIndex][numPlayers]) {
      secondMax = maxIndex;
      maxIndex = i;
    } else if (cardFrequencies[i][numPlayers] >
               cardFrequencies[secondMax][numPlayers]) {
      secondMax = i;
    }
  }
  if (cardFrequencies[maxIndex][numPlayers] >= 0.999f) {
    std::vector<int> ownedSuspects;
    for (int i = 0; i < 6; i++) {
      if (cards->contains(i)) ownedSuspects.push_back(i);
    }
    if (!ownedSuspects.empty()) {
      std::vector<int> shuffledSuspects = ownedSuspects;
      thread_local std::mt19937 rng(std::random_device{}());
      std::shuffle(shuffledSuspects.begin(), shuffledSuspects.end(), rng);
      
      int bestSuspect = shuffledSuspects[0];
      int minShown = timesShown[bestSuspect];
      for (int s : shuffledSuspects) {
        if (timesShown[s] < minShown) {
          minShown = timesShown[s];
          bestSuspect = s;
        }
      }
      probSus.push_back(bestSuspect);
    } else {
      probSus.push_back(maxIndex);
    }
  } else {
    probSus.push_back(maxIndex);
    probSus.push_back(secondMax);
  }

  maxIndex = 10;
  secondMax = 11;
  for (int i = 10; i < 16; i++) {
    if (cardFrequencies[i][numPlayers] >
        cardFrequencies[maxIndex][numPlayers]) {
      secondMax = maxIndex;
      maxIndex = i;
    } else if (cardFrequencies[i][numPlayers] >
               cardFrequencies[secondMax][numPlayers]) {
      secondMax = i;
    }
  }
  if (cardFrequencies[maxIndex][numPlayers] >= 0.999f) {
    std::vector<int> ownedWeapons;
    for (int i = 10; i < 16; i++) {
      if (cards->contains(i)) ownedWeapons.push_back(i);
    }
    if (!ownedWeapons.empty()) {
      std::vector<int> shuffledWeapons = ownedWeapons;
      thread_local std::mt19937 rng(std::random_device{}());
      std::shuffle(shuffledWeapons.begin(), shuffledWeapons.end(), rng);
      
      int bestWeapon = shuffledWeapons[0];
      int minShown = timesShown[bestWeapon];
      for (int w : shuffledWeapons) {
        if (timesShown[w] < minShown) {
          minShown = timesShown[w];
          bestWeapon = w;
        }
      }
      probWeapon.push_back(bestWeapon);
    } else {
      probWeapon.push_back(maxIndex);
    }
  } else {
    probWeapon.push_back(maxIndex);
    probWeapon.push_back(secondMax);
  }
  int maxRoom = 20;
  for (int i = 20; i < 29; i++) {
    if (cardFrequencies[i][numPlayers] > cardFrequencies[maxRoom][numPlayers]) {
      maxRoom = i;
    }
  }

  if (cardFrequencies[maxRoom][numPlayers] >= 0.999f) {
    std::vector<int> ownedPossibleRooms;
    for (int r : possibleRooms) {
      if (cards->contains(r + 20)) ownedPossibleRooms.push_back(r);
    }
    if (!ownedPossibleRooms.empty()) {
      std::vector<int> shuffledRooms = ownedPossibleRooms;
      thread_local std::mt19937 rng(std::random_device{}());
      std::shuffle(shuffledRooms.begin(), shuffledRooms.end(), rng);
      
      int bestRoom = shuffledRooms[0];
      int minShown = timesShown[bestRoom + 20];
      for (int r : shuffledRooms) {
        if (timesShown[r + 20] < minShown) {
          minShown = timesShown[r + 20];
          bestRoom = r;
        }
      }
      possibleRooms.clear();
      possibleRooms.push_back(bestRoom);
    }
  }

  std::vector<std::vector<float>> tempMatrix = cardFrequencies;

  float bestEntropy = 9999.0f;
  float bestScore = 9999.0f;
  Suggestion bestSuggestion = {-1, -1, -1};

  for (int s : probSus) {
    for (int w : probWeapon) {
      for (int r : possibleRooms) {
        float expectedEntropyForGuess = 0.0f;
        float totalProbWeight = 0.0f;

        int suggestedCards[3] = {s, w, r + 20}; // Use absolute room card ID

        // Loop Through All The Players

        for (int p = 0; p < numPlayers; p++) {
          if (p == playerIndex) {
            continue; // You cant show a card to youself
          }
          for (int card : suggestedCards) {
            float originalProb = cardFrequencies[card][p];
            if (originalProb < 0.0001f) {
              // This player cant have this card so continue
              continue;
            }
            // Simulate, what if player p had this card
            std::fill(tempMatrix[card].begin(), tempMatrix[card].end(),
                      0.0f); // Assign all players a 0 prob of having this card
            tempMatrix[card][p] =
                1.0f; // Give player p a 100% chance of having that card

            float hypotheticalEntropy = calculateBoardEntropy(tempMatrix);
            expectedEntropyForGuess +=
                originalProb *
                hypotheticalEntropy; // weight this future by how likely it is
            totalProbWeight += originalProb;

            tempMatrix[card] = cardFrequencies[card]; // reset it
          }
        }

        // Normalize Expected Entropy

        if (totalProbWeight > 0.0001f) {
          expectedEntropyForGuess /= totalProbWeight;
        } else {
          expectedEntropyForGuess = calculateBoardEntropy(cardFrequencies);
        }

        if (comDebugMode) {
          std::cout << "    Candidate: Suggest " << numToCard(s, 0) << " with "
                    << numToCard(w - 10, 1) << " in " << numToCard(r, 2)
                    << " -> Expected Board Entropy: " << expectedEntropyForGuess
                    << std::endl;
        }

        // Select the Winner with heuristic bias towards cards with higher
        // envelope probability
        float suspectEnvProb = cardFrequencies[s][numPlayers];
        float weaponEnvProb = cardFrequencies[w][numPlayers];
        float roomEnvProb = cardFrequencies[r + 20][numPlayers];
        float score = expectedEntropyForGuess -
                      (suspectEnvProb + weaponEnvProb + roomEnvProb) * 2.0f;
        if (score < bestScore) {
          bestScore = score;
          bestEntropy = expectedEntropyForGuess;
          bestSuggestion = {s, w, r + 20}; // Store absolute room card ID
        }
      }
    }
  }

  if (comDebugMode) {
    std::cout << "  ==> SELECTED BEST: Suggest "
              << numToCard(bestSuggestion.suspect, 0) << " with "
              << numToCard(bestSuggestion.weapon - 10, 1) << " in "
              << numToCard(bestSuggestion.room - 20, 2)
              << " (Expected Entropy: " << bestEntropy << ")" << std::endl;
    std::cout
        << "---------------------------------------------------------\033[0m"
        << std::endl;
  }

  return bestSuggestion;
}

float Player::calculateBoardEntropy(
    const std::vector<std::vector<float>> &matrix) {
  float totalEntropy = 0.0;
  for (int card = 0; card < 29; card++) {
    if ((card >= 6 && card <= 9) || (card >= 16 && card <= 19))
      continue;

    for (int loc = 0; loc <= numPlayers; loc++) {
      float p = matrix[card][loc];
      if (p > 0.0001f) {
        totalEntropy += -p * std::log2(p);
      }
    }
  }
  return totalEntropy;
}

bool Player::checkAccusationReady(Suggestion &accusation) {
  std::vector<std::vector<float>> cardFrequencies =
      cardMonteCarlo(ClueData::MONTE_CARLO_ITERATIONS);

  int bestSuspect = -1;
  int bestWeapon = -1;
  int bestRoom = -1;

  // check envelope

  for (int s = 0; s < 6; s++) {
    if (cardFrequencies[s][numPlayers] >= 0.999f) {
      bestSuspect = s;
      break;
    }
  }
  for (int w = 10; w < 16; w++) {
    if (cardFrequencies[w][numPlayers] >= 0.999f) {
      bestWeapon = w;
      break;
    }
  }
  for (int r = 20; r < 29; r++) {
    if (cardFrequencies[r][numPlayers] >= 0.999f) {
      bestRoom = r;
      break;
    }
  }

  if (comDebugMode) {
    std::cout << "\033[1;33m\n--- [DEBUG] COM Player " << (playerIndex + 1)
              << " Accusation Readiness check ---" << std::endl;
    std::cout << "  Best Suspect: "
              << (bestSuspect != -1 ? numToCard(bestSuspect, 0)
                                    : "None (No suspect >= 99.9% in envelope)")
              << std::endl;
    std::cout << "  Best Weapon:  "
              << (bestWeapon != -1 ? numToCard(bestWeapon - 10, 1)
                                   : "None (No weapon >= 99.9% in envelope)")
              << std::endl;
    std::cout << "  Best Room:    "
              << (bestRoom != -1 ? numToCard(bestRoom - 20, 2)
                                 : "None (No room >= 99.9% in envelope)")
              << std::endl;
    if (bestRoom != -1 && bestSuspect != -1 && bestWeapon != -1) {
      std::cout << "  ==> COM player " << (playerIndex + 1)
                << " is ready to ACCUSE!" << std::endl;
    } else {
      std::cout << "  ==> COM player " << (playerIndex + 1)
                << " is not ready to accuse yet." << std::endl;
    }
    std::cout
        << "---------------------------------------------------------\033[0m"
        << std::endl;
  }

  if (bestRoom != -1 && bestSuspect != -1 && bestWeapon != -1) {
    accusation.suspect = bestSuspect;
    accusation.weapon = bestWeapon;
    accusation.room = bestRoom;
    return true;
  }
  return false;
}

void Player::showHand() {
  for (int i = 0; i < cards->getSize(); i++) {
    int card = cards->get(i);
    std::string cardName = numToCard(card % 10, card / 10);
    logger.print(LogLevel::Player, " - " + cardName);
  }
}