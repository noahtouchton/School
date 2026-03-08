// Constants.h
#pragma once  // This prevents the file from being included more than once
#include <string>
#include <vector>

namespace ClueData {
    // Using const means these can't be accidentally changed during the game
    const std::vector<std::string> ROOMS = {
        "Hall", "Lounge", "Dining Room", "Kitchen", "Ballroom", "Conservatory", "Billiard Room", "Library", "Study"
    };

    const std::vector<std::string> WEAPONS = {
        "Knife", "Candlestick", "Revolver", "Rope", "Lead Pipe", "Wrench"
    };

    const std::vector<std::string> SUSPECTS = {
        "Col. Mustard", "Prof. Plum", "Mr. Green", "Mrs. Peacock", "Miss Scarlett",  "Mrs. White"
    };
}