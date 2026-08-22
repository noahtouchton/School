# verify_bot.py
import sys
import os

# Add bot directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bot'))

def test_imports():
    print("[RUN] Verifying bot modules syntax and imports...")
    try:
        import config
        print("[OK] config.py loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load config.py: {e}")
        return False

    try:
        import window_tracker
        print("[OK] window_tracker.py loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load window_tracker.py: {e}")
        return False

    try:
        import input_handler
        print("[OK] input_handler.py loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load input_handler.py: {e}")
        return False

    try:
        import vision_helper
        print("[OK] vision_helper.py loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load vision_helper.py: {e}")
        return False

    try:
        import fishing_mode
        print("[OK] fishing_mode.py loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load fishing_mode.py: {e}")
        return False

    try:
        import slaying_mode
        print("[OK] slaying_mode.py loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load slaying_mode.py: {e}")
        return False

    try:
        import mining_mode
        print("[OK] mining_mode.py loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load mining_mode.py: {e}")
        return False

    try:
        import captcha_solver
        print("[OK] captcha_solver.py loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load captcha_solver.py: {e}")
        return False

    try:
        import main_controller
        print("[OK] main_controller.py loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load main_controller.py: {e}")
        return False

    return True

def test_coordinate_calculations():
    print("\n[RUN] Testing relative-to-absolute coordinate translation...")
    
    # Mock pygetwindow before importing WindowTracker to bypass runtime error
    import pygetwindow as gw
    
    class MockWindow:
        def __init__(self, left, top, width, height, title):
            self.left = left
            self.top = top
            self.width = width
            self.height = height
            self.title = title
            
    gw.getWindowsWithTitle = lambda title: [MockWindow(100, 100, 800, 600, "Minecraft 1.20")]

    from window_tracker import WindowTracker
    tracker = WindowTracker()
    
    # Test point
    abs_pt = tracker.rel_to_abs_point(0.5, 0.5)
    expected_pt = (500, 400) # 100 + 0.5*800, 100 + 0.5*600
    if abs_pt == expected_pt:
        print(f"[OK] Point translation correct: {abs_pt}")
    else:
        print(f"[FAIL] Point translation mismatch: Got {abs_pt}, expected {expected_pt}")
        return False

    # Test rect
    abs_rect = tracker.rel_to_abs_rect(0.25, 0.25, 0.75, 0.75)
    expected_rect = (300, 250, 700, 550) # 100+200, 100+150, 100+600, 100+450
    if abs_rect == expected_rect:
        print(f"[OK] Rect translation correct: {abs_rect}")
    else:
        print(f"[FAIL] Rect translation mismatch: Got {abs_rect}, expected {expected_rect}")
        return False
        
    return True

def main():
    print("=== BOT SKELETON VERIFICATION ===\n")
    imports_ok = test_imports()
    coords_ok = test_coordinate_calculations()
    
    if imports_ok and coords_ok:
        print("\n[SUCCESS] All modules compiled and imported properly!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Fix syntax errors highlighted above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
