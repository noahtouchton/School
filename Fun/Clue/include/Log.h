#pragma once
#include <iostream>
#include <string>
#include <format>

// The hierarchy: Game (0) < Player (1) < COM (2) < Warn (3)
enum class LogLevel {
    Game,   
    Player, 
    COM,    
    Warn    
};

class Logger {
    private:
        LogLevel level; // This is the "ceiling" for what we are allowed to print

    public:
        // Constructor
        Logger(LogLevel logLevel) : level(logLevel) {}

        // Change the log level mid-game if needed
        void setLevel(LogLevel logLevel) {
            level = logLevel;
        }

        // Main print function
        void print(LogLevel msgLevel, const std::string& message) {
            
            // 1. THE FILTER
            // If the message is a higher level than the logger's ceiling, ignore it!
            // Example: Logger is COM. msgLevel is Warn. Warn > COM, so we bail out.
            if (msgLevel > level) {
                return; 
            }

            // 2. THE TAGS
            // If it made it past the filter, print it with the correct prefix
            if (msgLevel == LogLevel::Game) {
                std::cout << message << std::endl;
            } else if (msgLevel == LogLevel::Player) {
                std::cout << "[Player] " << message << std::endl;
            } else if (msgLevel == LogLevel::COM) {
                std::cout << "[COM] " << message << std::endl;
            } else if (msgLevel == LogLevel::Warn) {
                std::cerr << "[WARN] " << message << std::endl;
            }
        }

        // Board printer (Treated as Game level info)
        void printBoard(const char board[9][9]) {
            // Mute the board too if the logger is somehow set below Game level
            if (LogLevel::Game > level) return; 

            for(int i = 0; i < 9; i++) {
                for(int j = 0; j < 9; j++) {
                    std::cout << ' ' << board[i][j] << ' ';
                    
                    // Vertical pipes
                    if (j == 2 || j == 5) {
                        std::cout << "|";
                    }
                }
                
                // Horizontal lines
                if (i == 2 || i == 5) {
                    std::cout << "\n-----------------------------\n";
                } else {
                    std::cout << std::endl;
                }
            }
        }

        void printColoredBoard(int* isPossible){
                for(int i=0; i<3; i++) {
                    for(int j=0; j<3; j++) {
                        int pos = i*3 +j;
                        if(isPossible[pos]) {
                            std::cout << "\033[1;32m O \033[0m"; // Green O for possible moves
                        } else {
                            std::cout << " * "; // Regular * for impossible moves
                        }
                        
                        // Vertical pipes
                        if (j == 2) {
                            std::cout << "|";
                        }
                    }
                    
                    // Horizontal lines
                    if (i == 2) {
                        std::cout << "\n-----------------------------\n";
                    } else {
                        std::cout << std::endl;
                    }
                }
        }
};