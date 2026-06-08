# Clue Game Development Notes & Log

> [!IMPORTANT]
> **AI INSTRUCTIONS:** DO NOT change any code files or modify the codebase in any way unless explicitly asked to do so in that specific user request.

Welcome! This file (`gemini.md`) is a dedicated space to track the design, architecture, tasks, and notes for this command-line implementation of the classic board game **Clue**, developed in C++.

---

## 📌 Project Overview
This project is a CLI (command-line interface) implementation of the **Clue** board game. It supports a mix of human and computer (COM) players, with custom logic handling movement, rooms, suggestions, accusations, and card deduction.

### Key Components:
- **Game Engine** (`src/game.cpp`, `include/Game.h`): Handles the setup, player order, suggestion/accusation flows, and the primary game loop.
- **Player Logic** (`src/player.cpp`, `include/Player.h`, `include/Types.h`): Manages human and AI players, tracking their hand of cards, sheets, notes (known cards / anti-cards), and deduction status.
- **Board Layout** (`include/Board.h`): Represents the classic Clue map, handles player coordinates, pathfinding, and room access.
- **Utility Data Structures** (`include/LinkedList.h`): Custom templated linked list implementation used for managing cards and suggestion histories.

---

## 📂 Directory Structure

```text
Clue/
├── CMakeLists.txt      # Build configuration (min CMake 3.10)
├── todo.txt            # Quick task notes
├── gemini.md           # This developer guide & workspace log
├── include/            # C++ Header files
│   ├── Board.h         # Game board & coordinates
│   ├── Constants.h     # Game constants (weapons, rooms, suspects)
│   ├── Game.h          # Main game loop and engine
│   ├── LinkedList.h    # Custom Linked List implementation
│   ├── Log.h           # Logger & game history helper
│   ├── Player.h        # Player class & functions
│   └── Types.h         # Common data structs (Suggestions, PlayerData)
└── src/                # C++ Source files
    ├── game.cpp        # Game execution & loop implementation
    ├── main.cpp        # App entry point (player configuration)
    └── player.cpp      # Player operations & AI logic
```

---

## 🛠️ Build and Run Instructions

This project uses **CMake** for build configuration.

### Prerequisites:
- CMake 3.10 or higher
- A C++ standard compiler (GCC, Clang, or MSVC)

### Compilation:
To compile the project from the root directory:

```bash
# 1. Create a build directory
mkdir -p build
cd build

# 2. Generate build files
cmake ..

# 3. Compile the executable
make
```

### Running the Game:
After building, run the compiled binary:

```bash
./Clue
```

---

## 🎯 Development Roadmap

### Active Tasks:
- [ ] **Player Database Persistence**: Implement loading and saving of player profile/history databases (noted in `todo.txt`).
- [ ] **AI Deduction Enhancements**: Improve the AI's deduction algorithms (`PlayerData::possibleSuspects` processing) to make smarter suggestions.
- [ ] **Game State Logs**: Improve `Log.h` tracking for debugging AI decisions.

### Completed Milestones:
- [x] Basic game loop and player configuration setup.
- [x] Room movement, suggestions, and card disproving mechanism.
- [x] Custom linked list implementation.


DO NOT CHANGE MY CODE UNLESS I SPECIFICALLY ASK YOU TO!

