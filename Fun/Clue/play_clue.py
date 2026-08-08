import subprocess
import select
import sys
import os
import re

def main():
    p = subprocess.Popen(["./build/Clue"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    
    suspects = ["Col. Mustard", "Prof. Plum", "Mr. Green", "Mrs. Peacock", "Miss Scarlett", "Mrs. White"]
    weapons = ["Knife", "Candlestick", "Revolver", "Rope", "Lead Pipe", "Wrench"]
    rooms = ["Hall", "Lounge", "Dining Room", "Kitchen", "Ballroom", "Conservatory", "Billiard Room", "Library", "Study"]
    
    possible_suspects = set(suspects)
    possible_weapons = set(weapons)
    possible_rooms = set(rooms)
    
    buf = ""
    target_room_sent = False
    current_try_room = 1
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    while True:
        rlist, _, _ = select.select([p.stdout], [], [], 0.05)
        if rlist:
            char_b = os.read(p.stdout.fileno(), 1)
            if not char_b:
                break
            char = char_b.decode("utf-8", errors="ignore")
            buf += char
            sys.stdout.write(char)
            sys.stdout.flush()
            
        clean_buf = ansi_escape.sub('', buf)
        
        if "Enable COM debugging mode? (1 for yes, 0 for no): " in clean_buf:
            p.stdin.write(b"0\n")
            buf = ""
        elif "Enter the number of human players (0-6): " in clean_buf:
            p.stdin.write(b"1\n")
            buf = ""
        elif "Enter the number of computer players (0-5): " in clean_buf:
            p.stdin.write(b"2\n")
            buf = ""
        elif "Here are your cards:" in clean_buf and "Press Enter to continue..." in clean_buf:
            for line in clean_buf.split("\n"):
                if " - " in line:
                    card = line.split(" - ")[1].strip()
                    possible_suspects.discard(card)
                    possible_weapons.discard(card)
                    possible_rooms.discard(card)
            p.stdin.write(b"\n")
            buf = ""
        elif "Press Enter to continue..." in clean_buf:
            if "showed you the card: " in clean_buf:
                for line in clean_buf.split("\n"):
                    if "showed you the card: " in line:
                        card = line.split("showed you the card: ")[1].strip()
                        possible_suspects.discard(card)
                        possible_weapons.discard(card)
                        possible_rooms.discard(card)
                        print(f"\n[BOT THOUGHT] Learned card: {card}. Remaining: {len(possible_suspects)} {len(possible_weapons)} {len(possible_rooms)}\n")
            p.stdin.write(b"\n")
            buf = ""
        elif "What room would you like to move to?" in clean_buf and not target_room_sent:
            current_try_room = 1
            print(f"[BOT SENDING] Room: {current_try_room}")
            p.stdin.write(f"{current_try_room}\n".encode())
            target_room_sent = True
            buf = ""
        elif "That is not a valid move" in clean_buf:
            current_try_room += 1
            if current_try_room > 9:
                current_try_room = 1
            print(f"[BOT SENDING] Fallback Room: {current_try_room}")
            p.stdin.write(f"{current_try_room}\n".encode())
            buf = ""
            target_room_sent = True
        elif "Enter a suspect to suggest:" in clean_buf:
            sus = list(possible_suspects)[0] if possible_suspects else suspects[0]
            print(f"[BOT SENDING] Suspect Suggestion: {sus}")
            p.stdin.write(f"{sus}\n".encode())
            buf = ""
            target_room_sent = False
        elif "Enter a weapon to suggest:" in clean_buf:
            wea = list(possible_weapons)[0] if possible_weapons else weapons[0]
            print(f"[BOT SENDING] Weapon Suggestion: {wea}")
            p.stdin.write(f"{wea}\n".encode())
            buf = ""
        elif "Are you ready to make an accusation?" in clean_buf:
            if len(possible_suspects) == 1 and len(possible_weapons) == 1 and len(possible_rooms) == 1:
                p.stdin.write(b"1\n")
            else:
                p.stdin.write(b"0\n")
            buf = ""
        elif "Enter a suspect for your accusation:" in clean_buf:
            sus = list(possible_suspects)[0] if possible_suspects else suspects[0]
            print(f"[BOT SENDING] Suspect Accusation: {sus}")
            p.stdin.write(f"{sus}\n".encode())
            buf = ""
        elif "Enter a weapon for your accusation:" in clean_buf:
            wea = list(possible_weapons)[0] if possible_weapons else weapons[0]
            print(f"[BOT SENDING] Weapon Accusation: {wea}")
            p.stdin.write(f"{wea}\n".encode())
            buf = ""
        elif "Enter a room for your accusation:" in clean_buf:
            rm = list(possible_rooms)[0] if possible_rooms else rooms[0]
            print(f"[BOT SENDING] Room Accusation: {rm}")
            p.stdin.write(f"{rm}\n".encode())
            buf = ""
        elif "Enter the number of the card you want to show" in clean_buf and "):" in clean_buf:
            p.stdin.write(b"1\n")
            buf = ""
        elif "Congratulations! Your accusation is correct" in clean_buf:
            print("\n\n[BOT] I WON!")
            break
        elif "is incorrect. You are eliminated from the game" in clean_buf:
            print("\n\n[BOT] I LOST AND WAS ELIMINATED.")
            break
        elif "wins the game!" in clean_buf:
            print("\n\n[BOT] A COM WON THE GAME.")
            break
        if p.poll() is not None:
            # Read remaining output
            rlist, _, _ = select.select([p.stdout], [], [], 0.1)
            while rlist:
                char_b = os.read(p.stdout.fileno(), 1024)
                if not char_b:
                    break
                sys.stdout.write(char_b.decode('utf-8', errors='ignore'))
                sys.stdout.flush()
                rlist, _, _ = select.select([p.stdout], [], [], 0.1)
            print(f"\n\n[BOT] Clue process exited with code: {p.poll()}")
            break

if __name__ == "__main__":
    main()
