#pragma once
#include <iostream>

enum class LogLevel {
    Game,
    Player,
    COM,
    Warn
};


class Logger {
    private:
        LogLevel level;

    public:
        Logger(LogLevel logLevel) : level(logLevel) {}

        void setLevel(LogLevel logLevel) {
            level = logLevel;
        }

        void print(LogLevel logLevel, const std::string& message) {
            if (logLevel >= level) {
                switch (logLevel) {
                    case LogLevel::Game:
                        std::cout << message << std::endl;
                        break;
                    case LogLevel::Player:
                        std::cout << "[PLAYER] " << message << std::endl;
                        break;
                    case LogLevel::COM:
                        std::cout << "[COM] " << message << std::endl;
                        break;
                    case LogLevel::Warn:
                        std::cout << "[WARN] " << message << std::endl;
                        break;
                }
            }
        }
        void printBoard(const char board[8][8]) {
            for(int i=0; i<8; i++) {
                for(int j=0; j<8; j++) {
                    std::cout<< ' ' << board[i][j] << ' ';
                    if (j == 2 || j == 5) {
                        std::cout << "|";
                    }
                }
                if (i == 2 || i == 5) {
                    std::cout << "\n-----------------------------\n";
                } else {
                    std::cout << std::endl;
                }
            }
        }


};