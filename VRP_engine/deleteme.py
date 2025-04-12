from pulp import LpProblem, LpVariable, LpMinimize, lpSum, COIN_CMD

# Define a simple problem
model = LpProblem("Test", LpMinimize)
x = LpVariable("x", 0, None)
y = LpVariable("y", 0, None)
model += x + y
model += x + 2 * y <= 10
model += 3 * x + y <= 12

# Solve using CBC
solver = COIN_CMD()
model.solve(solver=solver)
print("Status:", model.status)
print("Objective:", model.objective.value())