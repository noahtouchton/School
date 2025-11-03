import os, shutil, subprocess, tempfile
import pandas as pd

def run_xfoil_polar(*,
                    airfoil="NACA 4412",
                    Re=8e5, Mach=0.00, Ncrit=9,
                    a_start=-4.0, a_end=12.0, a_step=1.0,
                    iter_lim=400,
                    xfoil_path=None):
    def _find_xfoil():
        if xfoil_path and os.path.exists(xfoil_path):
            return xfoil_path
        envp = os.environ.get("XFOIL_PATH")
        if envp and os.path.exists(envp):
            return envp
        exe = shutil.which("xfoil")
        if exe:
            return exe
        raise FileNotFoundError("XFOIL not found. Pass xfoil_path=... or set XFOIL_PATH/add to PATH.")

    xfoil = _find_xfoil()
    tmpdir = tempfile.mkdtemp(prefix="xfoil_")
    polar_name = "polar.txt"
    polar_file = os.path.join(tmpdir, polar_name)
    cmd_file = os.path.join(tmpdir, "commands.txt")

    # IMPORTANT: exit VPAR with a blank line *before* ITER/PACC
    lines = []
    if isinstance(airfoil, str) and airfoil.strip().upper().startswith("NACA"):
        lines.append(airfoil.strip())
    else:
        lines += [f"LOAD {airfoil}", ""]   # accept default name

    lines += [
        "PANE",
        "OPER",
        f"VISC {Re}",
        f"MACH {Mach}",
        "VPAR",
        f"N {Ncrit}",
        "",                 # <-- exit VPAR submenu
        f"ITER {iter_lim}", # now back in OPER
        "PACC",
        polar_name,
        "",                 # start accumulation
        f"ASEQ {a_start} {a_end} {a_step}",
        "PACC",
        "",                 # close accumulation
        "QUIT"
    ]

    with open(cmd_file, "w", newline="\r\n") as f:
        f.write("\r\n".join(lines) + "\r\n")

    with open(cmd_file, "rb") as fin:
        proc = subprocess.run([xfoil], stdin=fin,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              cwd=tmpdir)

    if not os.path.exists(polar_file):
        print("=== XFOIL STDOUT ===\n", proc.stdout.decode(errors="ignore"))
        print("=== XFOIL STDERR ===\n", proc.stderr.decode(errors="ignore"))
        with open(cmd_file, "r", encoding="utf-8", errors="ignore") as cf:
            print("=== commands.txt ===\n", cf.read())
        raise RuntimeError("XFOIL did not produce a polar file (likely convergence/range issue).")

    rows = []
    with open(polar_file, "r", encoding="utf-8", errors="ignore") as f:
        started = False
        for s in f:
            s = s.strip()
            if not started:
                if s.startswith("----"):
                    started = True
                continue
            if not s:
                continue
            parts = s.split()
            if len(parts) >= 7:
                try:
                    a, cl, cd, cdp, cm, xtr_t, xtr_b = map(float, parts[:7])
                    rows.append((a, cl, cd, cdp, cm, xtr_t, xtr_b))
                except ValueError:
                    pass

    if not rows:
        raise RuntimeError("Parsed 0 rows from polar (try smaller alpha range or higher ITER).")

    
    return pd.DataFrame(rows, columns=["alpha","Cl","Cd","Cdp","Cm","Top_Xtr","Bot_Xtr"])
