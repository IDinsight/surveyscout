import numpy as np
from ortools.linear_solver import pywraplp


def min_target_optimization_model(
    cost_matrix,
    min_target,
    max_target,
    max_distance,
    max_total_distance,
):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    n_target, n_enum = cost_matrix.shape

    # Define binary variable matrix
    x = {}
    for i in range(n_target):
        for j in range(n_enum):
            x[i, j] = solver.BoolVar(f"x[{i},{j}]")

    # Add constraints

    for i in range(n_target):
        solver.Add(
            sum(x[i, j] for j in range(n_enum)) == 1
        )  # each target assigned to exactly 1 surveyor (surveyors can have any number > 0 of targets)
    # Min target constraint
    for j in range(n_enum):
        solver.Add(sum(x[i, j] for i in range(n_target)) >= min_target)
    # Max target constraint
    for j in range(n_enum):
        solver.Add(sum(x[i, j] for i in range(n_target)) <= max_target)

    for j in range(n_enum):
        solver.Add(
            sum(cost_matrix[i, j] * x[i, j] for i in range(n_target))
            <= max_total_distance
        )  # surveyor budget constraint

    for i in range(n_target):
        for j in range(n_enum):
            solver.Add(
                cost_matrix[i, j] * x[i, j] <= max_distance
            )  # surveyor budget constrain

    # Add objective
    solver.Minimize(
        sum(cost_matrix[i, j] * x[i, j] for i in range(n_target) for j in range(n_enum))
    )

    status = solver.Solve()

    # Check the result
    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal value: ", solver.Objective().Value())

        # Convert the solution to a matrix
        solution_matrix = np.zeros((n_target, n_enum))
        for i in range(n_target):
            for j in range(n_enum):
                solution_matrix[i, j] = x[i, j].solution_value()
        return solution_matrix
    else:
        return None
