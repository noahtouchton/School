import math
import sympy as sp

class Data:
    def __init__(self, name, var, value, uncertainty=0.0):
        self.name = name
        self.var = var
        self.value = value
        self.uncertainty = uncertainty

    def get_error(self):
        print(f"{self.value:.3f} ± {self.uncertainty:.3f}")



def create_dict(data_list):
    values = {}
    for d in data_list:
        key = d.var.name if isinstance(d.var, sp.Symbol) else str(d.var)
        values[key] = d.value
    return values


def RSS(name, value, function, data_list):
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
    try:
        expr = sp.sympify(rhs, locals=sym_map)
    except Exception as e:
        print(f"Could not parse RHS '{rhs}': {e}")
        return None

    val = 0.0
    for d in data_list:
        sym = d.var if isinstance(d.var, sp.Symbol) else sym_map[str(d.var)]
        dfdxi = sp.diff(expr, sym)
        dfdxi_val = float(dfdxi.subs(subs_map))
        val = val + (dfdxi_val * d.uncertainty)**2

    new_data = Data(str(name), lhs, value, math.sqrt(val))
    return new_data