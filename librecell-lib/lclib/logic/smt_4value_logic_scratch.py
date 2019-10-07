from pysmt.shortcuts import *

# Define 4-valued logic type.
#TL = Type('L')
#TH = Type('H')
#TX = Type('X')
#TZ = Type('Z')

#FVL = Type('FVL', arity=1)
FVL = Type('FVL', arity=0)

#env = get_env()
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
        Eq(f_not(L), H),
        Eq(f_not(H), L),
        Eq(f_not(X), X),
        Eq(f_not(Z), X),
        ]

solver = Solver()
for d in definitions:
    solver.add_assertion(d)

x = Symbol('x', FVL)
solver.add_assertion(Eq(x, f_not(H)))
solver.add_assertion(Eq(x, f_not(f_not(H))))
solver.add_assertion(Not(Eq(x, X)))
solver.add_assertion(Not(Eq(x, Z)))
sat = solver.check_sat()
print('sat =', sat)
model = solver.get_model()

#print(model[x])
