from pysmt.shortcuts import *

# Define 4-valued logic type.
#TL = Type('L')
#TH = Type('H')
#TX = Type('X')
#TZ = Type('Z')

FVLType = Type('FVL', arity=2)
# FVL = Type('FVL', arity=0)
# FVL = INT

env = get_env()
FVL = env.type_manager.get_type_instance(FVLType, BOOL, BOOL)
#L = env.type_manager.get_type_instance(FVL, TL)
#H = env.type_manager.get_type_instance(FVL, TH)
#X = env.type_manager.get_type_instance(FVL, TX)
#Z = env.type_manager.get_type_instance(FVL, TZ)

# Low
L = Symbol('L', FVL)
# High
H = Symbol('H', FVL)
# Unknown
X = Symbol('X', FVL)
# High-impedance
Z = Symbol('Z', FVL)


# Helper shortcut
Eq = EqualsOrIff

# Define boolean functions.
f_not = Symbol('f_not', FunctionType(FVL, [FVL]))

definitions = [
        # L,H,X,Z are all different.
        Not(Eq(L, H)),
        Not(Eq(L, X)),
        Not(Eq(L, Z)),
        Not(Eq(H, X)),
        Not(Eq(H, Z)),
        Not(Eq(X, Z)),

        # Define L,H,X,Z
        # Eq(L, Int(0)),
        # Eq(H, Int(1)),
        # Eq(X, Int(2)),
        # Eq(Z, Int(3)),

        # Define 'not' function.
        Eq(f_not(L), H),
        Eq(f_not(H), L),
        Eq(f_not(X), X),
        Eq(f_not(Z), X),
        ]

solver = Solver()
for d in definitions:
    solver.add_assertion(d)

print(FVL.as_smtlib())

x = Symbol('x', FVL)
solver.add_assertion(Eq(x, f_not(x)))
solver.add_assertion(Not(Eq(x, L)))
solver.add_assertion(Not(Eq(x, H)))
solver.add_assertion(Not(Eq(x, X)))
solver.add_assertion(Not(Eq(x, Z)))
# solver.add_assertion(Eq(x, f_not(f_not(H))))
# solver.add_assertion(Not(Eq(x, X)))
# solver.add_assertion(Not(Eq(x, Z)))
sat = solver.check_sat()

print('sat =', sat)
if sat:
        model = solver.get_model()
        # print(model)
        # print(model[x])
