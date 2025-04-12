from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary, LpContinuous, LpStatus, value, COIN_CMD
import random
from Customer import Customer


class MasterProblem:
    def __init__(self, problem, route_store, ips):
        self.problem = problem
        self.model = LpProblem(name="Master", sense=LpMinimize)
        self.route_store = route_store
        self.mip = False
        self.ips_stabilize = ips
        self.cov_cons = []
        self.solver = COIN_CMD()

    def add_variables(self):
        for num, route in self.route_store.routes.items():
            if self.mip:
                route.variable = LpVariable(name=f"Rt_{num}", cat=LpBinary)
            else:
                route.variable = LpVariable(name=f"Rt_{num}", lowBound=0, cat=LpContinuous)

    def set_objective(self):
        self.model += lpSum(route.variable * route.cost for route in self.route_store.routes.values())

    def coverage_constraint(self):
        for cust in self.problem.customers.values():
            if cust.is_depot:
                continue

            # Define the constraint
            constraint = lpSum(self.route_store.routes[x].variable for x in cust.in_routes) >= 1

            # Assign a unique name to the constraint
            constraint_name = f"Cust_{cust.cust_no}"

            # Add the constraint to the model with this name
            self.model += constraint, constraint_name

            # Assign the name to the Customer object for later reference
            cust.constraint_name = constraint_name

    def vehicle_constraint(self):
        vehicle_constraint = lpSum(x.variable for x in self.route_store.routes.values()) <= self.problem.vehicles.number
        self.model += vehicle_constraint, "Vehicles"

    def write_lp(self, filename=None):
        if filename is None:
            filename = "_lp.lp"
        self.model.writeLP(filename)

    def solve(self, lp_relaxed=False):
        self.model.solve(solver=self.solver)
        if lp_relaxed:
            for cust in self.problem.customers.values():
                if hasattr(cust, "constraint_name"):  # Check if constraint_name exists
                    cust.dual = value(self.model.constraints[cust.constraint_name].pi)

    def formulate_problem(self):
        self.add_variables()
        self.set_objective()
        self.coverage_constraint()

    def getAllSolComponents(self):
        x = [value(var) for var in self.model.variables()]
        slack = [self.model.constraints[name].slack for _, name in self.cov_cons]
        dual = [self.model.constraints[name].pi for _, name in self.cov_cons]
        dj = [var.dj for var in self.model.variables()]
        return x, slack, dual, dj

    def print_solution(self):
        solution = [value(var) for var in self.model.variables()]
        selected = [i for i in range(len(solution)) if solution[i] > 0.5]

        selected_rts = []
        for i in selected:
            rte = []
            for j in self.route_store.routes[i].stops:
                rte.append(j.customer.cust_no)
            selected_rts.append(rte)
            print(i, rte)
        return selected

    def ips_stabilization(self):
        if not self.ips_stabilize:
            return

        random.seed(a=1)
        x, slack, dual, df = self.getAllSolComponents()
        for i in range(len(dual)):
            self.problem.customers[i + 1].dual = 0

        R_star = [i for i in range(len(x)) if x[i] > 0]  # routes with positive x values
        Not_R_star = [i for i in range(len(x)) if x[i] <= 0]
        C = [i for i in range(len(slack)) if slack[i] > 0]  # constraints with non-zero slack
        Not_C = [i for i in range(len(slack)) if slack[i] <= 0]

        dual_vec = []
        for num in range(20):
            u = [random.random() for _ in range(self.problem.total_customers)]

            for k in Not_C:
                self.model.constraints[self.cov_cons[k][1]].constant = u[k]

            for k in C:
                self.model.constraints[self.cov_cons[k][1]].constant = 0

            for k in R_star:
                self.route_store.routes[k].variable.lowBound = 0

            for k in Not_R_star:
                self.route_store.routes[k].variable.lowBound = 0

            self.model.solve(solver=self.solver)

            x_p, slack_p, dual_p, df_p = self.getAllSolComponents()
            dual_vec.append(dual_p)

        avg_duals = [sum(x) / len(x) for x in zip(*dual_vec)]

        for ind, x in enumerate(avg_duals):
            self.problem.customers[ind + 1].dual = x

        print("Duals updated")