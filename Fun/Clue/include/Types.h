#pragma once
#include "LinkedList.h"
#include <vector>

struct Suggestion {
  int suspect;
  int weapon;
  int room;

  bool operator==(const Suggestion& other) const {
    return suspect == other.suspect && weapon == other.weapon && room == other.room;
  }

  bool operator!=(const Suggestion& other) const {
    return !(*this == other);
  }
};

struct Disprove {
  LinkedList<int> *possibleCards; // The specific card numbers that can disprove
                                  // the suggestion
  int player; // The player who can disprove the suggestion with this card

  Disprove()
      : possibleCards(nullptr), player(-1) {
  } // Default constructor for no disprove
};

class PlayerData {
public:
  int index;
  LinkedList<int> *cards;
  LinkedList<int> *antiCards; // Cards we know the player does not have
  LinkedList<Suggestion> *possibleSuspects;
  PlayerData(int idx)
      : index(idx), cards(new LinkedList<int>()),
        antiCards(new LinkedList<int>()),
        possibleSuspects(new LinkedList<Suggestion>()) {}

  ~PlayerData() {
    delete cards;
    delete antiCards;
    delete possibleSuspects;
  }
};

struct CardPossession {
  int card;
  int player; // 6 means its in the envelope
};



struct playerSuggestion {
  Suggestion suggestion;
  int player;
};