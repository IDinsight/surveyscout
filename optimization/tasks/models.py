import numpy as np
from ortools.linear_solver import pywraplp

from optimization.utils import get_percentile_distance


def min_target_optimization_model(
    cost_matrix,
    min_target,
    max_target,
    max_cost,
    max_total_cost,
):
    """
    Formulate and solve an optimization model to assign targets to surveyors while
    minimizing total cost subject to specific constraints.
    The following constraints are considered:
        - Each target is assigned to exactly one surveyor.
        - Each surveyor is assigned at least `min_target` targets and at most `max_target` targets.
        - The cost of assigning a surveyor to a target does not exceed `max_cost`.
        - The total cost of all assignments for a surveyor does not exceed `max_total_cost`.

    Parameters
    ----------
    cost_matrix : array_like
        A 2D array representing the costs of assigning surveyors to targets,
        typically costs. Shape: (n_target, n_enum).

    min_target : int
        The minimum number of targets each enumerator is required to visit.

    max_target : int
        The maximum number of targets each enumerator is allowed to visit.

    max_cost : float
        The maximum allowable cost to travel to a single target.

    max_total_cost : float
        The maximum total cost to travel to visit targets.

    Returns
    -------
    solution_matrix : array_like or None
        An array representing the solution matrix where each element indicates the
        assignment of targets to surveyors. If the problem is not solvable,
        None is returned.

    See Also
    --------
    pywraplp.Solver : The underlying solver used from OR-Tools for optimization.
    """

    solver = pywraplp.Solver.CreateSolver("SCIP")
    n_target, n_enum = cost_matrix.shape

    x = {}
    for i in range(n_target):
        for j in range(n_enum):
            x[i, j] = solver.BoolVar(f"x[{i},{j}]")

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
            sum(cost_matrix[i, j] * x[i, j] for i in range(n_target)) <= max_total_cost
        )  # surveyor budget constraint

    for i in range(n_target):
        for j in range(n_enum):
            solver.Add(
                cost_matrix[i, j] * x[i, j] <= max_cost
            )  # surveyor budget constraint

    solver.Minimize(
        sum(cost_matrix[i, j] * x[i, j] for i in range(n_target) for j in range(n_enum))
    )

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal value: ", solver.Objective().Value())

        solution_matrix = np.zeros((n_target, n_enum))
        for i in range(n_target):
            for j in range(n_enum):
                solution_matrix[i, j] = x[i, j].solution_value()
        return solution_matrix
    else:
        return None


def recursive_min_target_optimization(
    cost_matrix,
    min_target,
    max_target,
    max_cost,
    max_total_cost,
    param_increment=5,
):
    """
    Recursively optimize the minimum targeting constraints using the `min_target_optimization_model`
    model, adjusting parameters incrementally until a solution is found
    or the minimum number of targets parameter reaches zero.

    The function attempts to solve the optimization problem with initially provided
    constraints. If the solution is not found,it incrementally alters the `min_target`, `max_target`,
    `max_cost`, and `max_total_cost` parameters by a given percentage defined by `param_increment`
    until a solution is reached or `min_target` is zero.

    Parameters
    ----------
    cost_matrix : numpy.ndarray
        A 2D array representing the costs of assigning surveyors to targets,
        typically costs. Shape: (n_target, n_enum).

    min_target : int
        The minimum number of targets each enumerator is required to visit.

    max_target : int
        The maximum number of targets each enumerator is allowed to visit.

    max_cost : float
        The initial maximum cost assignable to a surveyor to visit a single target.

    max_total_cost : float
        The initial maximum total cost assignable to a surveyor.

    param_increment : int, optional
        The value by which the parameter bounds and percentiles are adjusted during the
        recursion if no solution is found (default is 5).

    Returns
    -------
    tuple
        A tuple containing the solution matrix with the assigned targets to surveyors,
        and a dictionary of the parameters used to find the solution. If no solution
        is found, returns (None, empty dictionary).
    """

    result = min_target_optimization_model(
        cost_matrix,
        min_target,
        max_target,
        max_cost,
        max_total_cost,
    )

    if result is not None:
        params = {
            "min_target": min_target,
            "max_target": max_target,
            "max_cost": max_cost,
            "max_total_cost": max_total_cost,
        }

        return result, params

    elif min_target > 0:
        return recursive_min_target_optimization(
            cost_matrix,
            min_target=min_target * (1 - param_increment / 100),
            max_target=max_target * (1 + param_increment / 100),
            max_cost=max_cost * (1 + param_increment / 100),
            max_total_cost=max_total_cost * (1 + param_increment / 100),
        )
    else:
        return None, dict({})
