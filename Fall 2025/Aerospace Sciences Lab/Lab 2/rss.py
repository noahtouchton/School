import math
import sympy as sp

class Data:
    def __init__(self, name, var, value, uncertainty=0.0):
        self.name = name
        self.var = var
        self.value = value
        self.uncertainty = uncertainty

    def get_error(self):
        print(f"{self.value:.3e} ± {self.uncertainty:.3e}")

    def get_description(self):
        print(f"{self.name}, {self.var}: {self.value:.3e} ± {self.uncertainty:.3e}")



def create_dict(data_list):
    values = {}
    for d in data_list:
        key = d.var.name if isinstance(d.var, sp.Symbol) else str(d.var)
        values[key] = d.value
    return values


def RSS(name, function, data_list):
    #allow the user to input spaces but remove them here
    function = function.replace(" ", "")
    if "=" not in function:
        print("Please enter a valid function: expected an '='.")
        return None

    lhs, rhs = function.split("=", 1)

    sym_map = {}  # name -> Symbol
    subs_map = {}  # Symbol -> numeric value
    for d in data_list:
        if isinstance(d.var, sp.Symbol):
            sym = d.var
            var_name = sym.name
        else:
            var_name = str(d.var)
            sym = sp.Symbol(var_name)
        sym_map[var_name] = sym
        subs_map[sym] = d.value

    lhs_sym = sp.Symbol(lhs)

    try:
        expr = sp.sympify(rhs, locals=sym_map)
    except Exception as e:
        print(f"Could not parse RHS '{rhs}': {e}")
        return None
    
    missing = [s for s in expr.free_symbols if s not in subs_map]
    if missing:
        names = ", ".join(sorted([s.name for s in missing]))
        print(f"Missing values for : {names}")
        return None

    try:
        nominal_value = float(expr.subs(subs_map))
    except Exception as e:
        print(f"Could not evaluate expression '{rhs_str}': {e}")
        return None

    rss_sq = 0.0
    for d in data_list:
        sym = d.var if isinstance(d.var, sp.Symbol) else sym_map[str(d.var)]
        dfdxi = sp.diff(expr, sym)
        dfdxi_val = float(dfdxi.subs(subs_map))
        rss_sq = rss_sq + (dfdxi_val * d.uncertainty)**2

    new_data = Data(str(name), lhs, nominal_value, math.sqrt(rss_sq))
    return new_data