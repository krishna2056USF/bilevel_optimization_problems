"""Reusable educational implementation of Algorithm 7 (DFS branch-and-cut).

Modes:
  prop97 : Proposition 9.7 only (Algorithm 7 cut step)
  hybrid : Proposition 9.9 if refinement infeasible, else Proposition 9.7
  prop99 : Proposition 9.9 if applicable; otherwise branch on an unfixed
           binary linking variable (explicit deviation from Algorithm 7 line 20,
           allowed by the text after Algorithm 7).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from math import ceil, floor, inf, isfinite
from time import monotonic
from typing import Optional
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import numpy as np

TOL = 1e-6


@dataclass
class Node:
    number: int
    model: gp.Model
    parent: Optional[int] = None
    depth: int = 0
    branch: str = "root"
    fixed_links: frozenset[str] = frozenset()
    lp_bound: Optional[float] = None
    solution: Optional[dict[str, float]] = None
    status: str = "pending"
    cuts: int = 0


class BilevelProblem:
    """Adapter contract: subclass and override root_lp(), follower(), refinement().

    Root model variables must be named x[0]..x[nx-1], y[0]..y[ny-1].
    All vars listed in integer_names must have integer values in an original
    feasible solution. Additional continuous variables are allowed, but the
    Prop. 9.7 implementation below requires ALL LP variables to be integer.

    Follower returns (follower_optimum_in_original_sense, y_vector).
    Refinement returns (minimized_transformed_leader_obj, x_vector, y_vector)
    or None only if the refinement MILP was PROVEN infeasible.
    """

    name = "generic"
    nx = 0
    ny = 0
    linking_names: tuple[str, ...] = ()
    integer_names: tuple[str, ...] = ()
    follower_sense = GRB.MAXIMIZE

    def root_lp(self) -> gp.Model:
        raise NotImplementedError

    def follower(self, x: list[int]) -> tuple[float, list[int]]:
        raise NotImplementedError

    def refinement(self, x: list[int], phi: float):
        raise NotImplementedError

    def candidate_follower_value(self, x: list[int], sol: dict[str, float]) -> float:
        raise NotImplementedError

    def original_leader_obj(self, transformed_value: float) -> float:
        return transformed_value

    def description(self, x: list[int], y: list[int]) -> str:
        return f"x={x}, y={y}"

    def check_follower_optimal(self, x, sol, phi, tol=TOL) -> bool:
        val = self.candidate_follower_value(x, sol)
        if self.follower_sense == GRB.MAXIMIZE:
            return val >= phi - tol
        return val <= phi + tol


class Knapsack915(BilevelProblem):
    name = "Exercise 9.15"
    nx = ny = 3
    linking_names = tuple(f"x[{i}]" for i in range(3))
    integer_names = tuple(f"x[{i}]" for i in range(3)) + tuple(f"y[{i}]" for i in range(3))
    follower_sense = GRB.MAXIMIZE

    def root_lp(self):
        m = gp.Model("knapsack915_lp")
        x = m.addVars(3, lb=0, ub=1, name="x")
        y = m.addVars(3, lb=0, ub=1, name="y")
        m.setObjective(4*y[0] + 3*y[1] + 3*y[2], GRB.MINIMIZE)
        m.addConstr(2*x[0]+x[1]+x[2] <= 2, name="leader_budget")
        m.addConstr(4*y[0]+3*y[1]+2*y[2] <= 4, name="follower_capacity")
        for i in range(3):
            m.addConstr(x[i]+y[i] <= 1, name=f"interdiction_{i}")
        m.update()
        return m

    def follower(self, x):
        m = gp.Model("follower_915")
        m.Params.OutputFlag = 0
        y = m.addVars(3, vtype=GRB.BINARY, name="yf")
        obj = 4*y[0]+3*y[1]+3*y[2]
        m.setObjective(obj, GRB.MAXIMIZE)
        m.addConstr(4*y[0]+3*y[1]+2*y[2] <= 4)
        for i in range(3):
            m.addConstr(y[i] <= 1-x[i])
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Follower failed: status {m.Status}")
        return m.ObjVal, [int(round(y[i].X)) for i in range(3)]

    def refinement(self, x, phi):
        m = gp.Model("refinement_915")
        m.Params.OutputFlag = 0
        y = m.addVars(3, vtype=GRB.BINARY, name="yr")
        obj = 4*y[0]+3*y[1]+3*y[2]
        m.setObjective(obj, GRB.MINIMIZE)
        m.addConstr(4*y[0]+3*y[1]+2*y[2] <= 4)
        for i in range(3):
            m.addConstr(y[i] <= 1-x[i])
        m.addConstr(obj >= phi-TOL)
        m.optimize()
        if m.Status == GRB.INFEASIBLE:
            return None
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Refinement failed: {m.Status}")
        return m.ObjVal, list(x), [int(round(y[i].X)) for i in range(3)]

    def candidate_follower_value(self, x, sol):
        return sum(w*sol[f"y[{i}]"] for i, w in enumerate((4,3,3)))


PATH_ADJ = {
    1:[2,3,4,5], 2:[6,7,3], 5:[8,9,4],
    3:[2,4,6,7,8], 4:[3,5,7,8,9],
    6:[10,11,7], 9:[12,13,8], 7:[6,8,10,11,12],
    8:[7,9,11,12,13], 10:[14,15,11],
    13:[16,17,12], 11:[10,12,14,15,16],
    12:[11,13,15,16,17], 14:[18,15],
    17:[18,16], 15:[18,14,16], 16:[18,15,17], 18:[]
}


class PathInterdiction(BilevelProblem):
    """HW1 path interdiction, binary shortest-path arc variables.

    variant='objective': cost on interdicted arcs increases from c_e to c_e+q_e,
      q_e=c_e in the uploaded notebook. Linearize x_e*y_e as auxiliary w_e.
    variant='constraint': interdicted arcs cannot be used: x_e+y_e<=1.
    In both variants the leader MAXIMIZES the follower's MINIMUM path cost,
    so node objective is its negative, to fit minimization Algorithm 7.
    """

    follower_sense = GRB.MINIMIZE

    def __init__(self, variant="objective", budget=3):
        if variant not in ("objective", "constraint"):
            raise ValueError("variant must be 'objective' or 'constraint'")
        self.variant = variant
        self.name = f"HW1 Problem 6 ({variant} interdiction)"
        self.budget = int(budget)
        self.vertices = list(PATH_ADJ)
        self.edges = [(i,j) for i in self.vertices for j in PATH_ADJ[i]]
        self.eligible = [e for e in self.edges if e[0] != 1 and e[1] != 18]
        self.x_index = {e:i for i,e in enumerate(self.eligible)}
        self.y_index = {e:i for i,e in enumerate(self.edges)}
        self.cost = {e:sum(e) for e in self.edges}
        self.q = {e:self.cost[e] for e in self.eligible}
        self.nx = len(self.eligible)
        self.ny = len(self.edges)
        self.linking_names = tuple(f"x[{i}]" for i in range(self.nx))
        aux = tuple(f"w[{i}]" for i in range(self.nx)) if variant == "objective" else ()
        self.integer_names = (self.linking_names +
                              tuple(f"y[{i}]" for i in range(self.ny)) + aux)
        self.incoming = {v:[self.y_index[e] for e in self.edges if e[1]==v]
                         for v in self.vertices}
        self.outgoing = {v:[self.y_index[e] for e in self.edges if e[0]==v]
                         for v in self.vertices}

    def _create(self, discrete: bool, x_fixed=None, phi=None):
        m = gp.Model("path_refinement" if discrete else "path_node_lp")
        m.Params.OutputFlag = 0
        xt = GRB.BINARY if discrete else GRB.CONTINUOUS
        yt = GRB.BINARY if discrete else GRB.CONTINUOUS
        x = m.addVars(self.nx, lb=0, ub=1, vtype=xt, name="x")
        y = m.addVars(self.ny, lb=0, ub=1, vtype=yt, name="y")
        m.addConstr(gp.quicksum(x[i] for i in range(self.nx)) <= self.budget,
                    name="interdiction_budget")
        for v in self.vertices:
            rhs = 1 if v==1 else (-1 if v==18 else 0)
            m.addConstr(gp.quicksum(y[i] for i in self.outgoing[v]) -
                        gp.quicksum(y[i] for i in self.incoming[v]) == rhs,
                        name=f"flow_{v}")
        normal_cost = gp.quicksum(self.cost[e]*y[j] for j,e in enumerate(self.edges))
        if self.variant == "constraint":
            for i,e in enumerate(self.eligible):
                m.addConstr(x[i]+y[self.y_index[e]] <= 1, name=f"removed_{i}")
            follower_cost = normal_cost
        else:
            # w = x*y, exact for binary variables and its convex hull when relaxed
            w = m.addVars(self.nx, lb=0, ub=1, vtype=xt, name="w")
            for i,e in enumerate(self.eligible):
                ye = y[self.y_index[e]]
                m.addConstr(w[i] <= x[i], name=f"prod_x_{i}")
                m.addConstr(w[i] <= ye, name=f"prod_y_{i}")
                m.addConstr(w[i] >= x[i]+ye-1, name=f"prod_lb_{i}")
            follower_cost = normal_cost + gp.quicksum(
                self.q[e]*w[i] for i,e in enumerate(self.eligible))
        m.setObjective(-follower_cost, GRB.MINIMIZE)
        if x_fixed is not None:
            for i,val in enumerate(x_fixed):
                m.addConstr(x[i]==val, name=f"fixed_x_{i}")
            if phi is not None:
                # follower MINIMIZES path cost, so require cost <= optimal phi
                m.addConstr(follower_cost <= phi + TOL, name="follower_optimality")
        m.update()
        return m

    def root_lp(self):
        return self._create(discrete=False)

    def follower(self, x):
        m = gp.Model("path_follower")
        m.Params.OutputFlag = 0
        y = m.addVars(self.ny, vtype=GRB.BINARY, name="yf")
        for v in self.vertices:
            rhs = 1 if v==1 else (-1 if v==18 else 0)
            m.addConstr(gp.quicksum(y[i] for i in self.outgoing[v]) -
                        gp.quicksum(y[i] for i in self.incoming[v]) == rhs)
        if self.variant == "constraint":
            for i,e in enumerate(self.eligible):
                m.addConstr(y[self.y_index[e]] <= 1-x[i])
        expr = gp.quicksum(
            (self.cost[e] + (self.q[e]*x[self.x_index[e]]
             if self.variant=="objective" and e in self.x_index else 0))*y[j]
            for j,e in enumerate(self.edges))
        m.setObjective(expr, GRB.MINIMIZE)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Follower status {m.Status}: a leader decision may disconnect the network")
        return m.ObjVal, [int(round(y[j].X)) for j in range(self.ny)]

    def refinement(self, x, phi):
        m = self._create(discrete=True, x_fixed=x, phi=phi)
        m.optimize()
        if m.Status == GRB.INFEASIBLE:
            return None
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Refinement failed: {m.Status}")
        return (m.ObjVal, list(x),
                [int(round(m.getVarByName(f"y[{j}]").X)) for j in range(self.ny)])

    def candidate_follower_value(self, x, sol):
        # Evaluate follower's actual objective at the (binary) (x, y) pair.
        val = sum(self.cost[e]*sol[f"y[{j}]"] for j,e in enumerate(self.edges))
        if self.variant=="objective":
            val += sum(self.q[e]*x[i]*sol[f"y[{self.y_index[e]}]"]
                       for i,e in enumerate(self.eligible))
        return val

    def original_leader_obj(self, transformed_value):
        return -transformed_value

    def description(self, x, y):
        interdicted = [e for i,e in enumerate(self.eligible) if x[i]]
        used = [e for j,e in enumerate(self.edges) if y[j]]
        return f"interdicted={interdicted}; selected-path-arcs={used}"


class BranchAndCut:
    def __init__(self, problem: BilevelProblem, cut_strategy="prop97",
                 max_nodes=10000, max_cuts=20000, time_limit=None,
                 verbose=True):
        if cut_strategy not in ("prop97", "prop99", "hybrid"):
            raise ValueError("cut_strategy must be prop97, prop99 or hybrid")
        self.problem = problem
        self.cut_strategy = cut_strategy
        self.max_nodes = max_nodes
        self.max_cuts = max_cuts
        self.time_limit = time_limit
        self.verbose = verbose
        self.nodes: dict[int,Node] = {}
        self.stack: list[int] = []
        self.next_number = 0
        self.UB = inf  # All node objectives expressed as minimization
        self.incumbent = None
        self.cut_count = 0
        self.lp_count = 0
        self.status = "NOT_STARTED"

    def log(self, message):
        if self.verbose:
            print(message)

    def _check_time(self, start):
        if self.time_limit is not None and monotonic()-start > self.time_limit:
            raise TimeoutError("Wall-clock limit reached; optimality NOT certified")

    def _children(self, node, varname, split, linking=False):
        node.model.update()
        ml = node.model.copy()
        mr = node.model.copy()
        ml.addConstr(ml.getVarByName(varname) <= split[0],
                     name=f"branch_left_{self.next_number}")
        mr.addConstr(mr.getVarByName(varname) >= split[1],
                     name=f"branch_right_{self.next_number+1}")
        fixed = set(node.fixed_links)
        if varname in self.problem.linking_names:
            fixed.add(varname)
        left = Node(self.next_number, ml, node.number, node.depth+1,
                    f"{varname}<={split[0]}", frozenset(fixed))
        right = Node(self.next_number+1, mr, node.number, node.depth+1,
                     f"{varname}>={split[1]}", frozenset(fixed))
        self.next_number += 2
        self.nodes[left.number] = left
        self.nodes[right.number] = right
        self.stack.extend([right.number,left.number]) # LIFO processes left first
        node.status = "branch_linking" if linking else "branch_fractional"
        self.log(f"  {node.status}: {varname} -> {left.number}, {right.number}")

    def _add_prop97(self, node):
        """Local Proposition 9.7 inequality at an integer LP vertex.

        All active rows are normalized to >=, and all variable bounds are
        included if tight. Equalities are treated as >= (one of two valid
        orientations). Integral rows + full rank isolate a unique integer point.
        """
        m = node.model
        vs = m.getVars()
        n = len(vs)
        if set(v.VarName for v in vs) != set(self.problem.integer_names):
            raise ValueError("Prop9.7 requires all node LP variables integer in this implementation")
        vec = np.array([v.X for v in vs])
        if any(abs(v-round(v))>TOL for v in vec):
            raise ValueError("Prop9.7 requires an integer candidate")
        idx = {v.VarName:i for i,v in enumerate(vs)}
        rows, rhs, names = [], [], []
        for constr in m.getConstrs():
            if abs(constr.Slack) > TOL:
                continue
            a = np.zeros(n)
            expr = m.getRow(constr)
            for j in range(expr.size()):
                a[idx[expr.getVar(j).VarName]] += expr.getCoeff(j)
            # equality can be represented by either orientation; >= is valid
            sign = -1 if constr.Sense=='<' else 1
            rows.append(sign*a)
            rhs.append(sign*constr.RHS)
            names.append(constr.ConstrName)
        for i,v in enumerate(vs):
            e=np.zeros(n);e[i]=1
            if v.LB > -GRB.INFINITY and abs(v.X-v.LB)<=TOL:
                rows.append(e.copy()); rhs.append(v.LB); names.append(v.VarName+"_LB")
            if v.UB < GRB.INFINITY and abs(v.X-v.UB)<=TOL:
                rows.append(-e.copy()); rhs.append(-v.UB); names.append(v.VarName+"_UB")
        if not rows or np.linalg.matrix_rank(np.array(rows), tol=1e-8) != n:
            raise ValueError("Prop9.7 cannot certify an integer basic vertex; cut not added")
        if any(np.max(np.abs(a-np.rint(a)))>TOL or abs(b-round(b))>TOL
               for a,b in zip(rows,rhs)):
            raise ValueError("Prop9.7 requires integer-valued active rows and RHS")
        coeff = np.rint(np.sum(rows,axis=0)).astype(np.int64)
        bound = int(round(sum(rhs)))+1
        if coeff @ np.rint(vec).astype(np.int64) >= bound:
            raise ValueError("Proposition 9.7 failed to separate candidate")
        m.addConstr(gp.quicksum(int(coeff[i])*vs[i] for i in range(n)) >= bound,
                    name=f"prop97_cut_{self.cut_count+1}")
        m.update()
        self.cut_count += 1
        node.cuts += 1
        nonzero = {vs[i].VarName:int(coeff[i]) for i in range(n) if coeff[i]}
        self.log(f"  Prop 9.7 cut {self.cut_count}: {nonzero} >= {bound}; active={names}")

    def _add_prop99(self, node, x):
        # Caller MUST already have established infeasible refinement.
        m=node.model
        expr = gp.quicksum((1-m.getVarByName(f"x[{i}]")) if x[i] else
                           m.getVarByName(f"x[{i}]") for i in range(self.problem.nx))
        m.addConstr(expr>=1,name=f"prop99_cut_{self.cut_count+1}")
        m.update()
        self.cut_count += 1
        node.cuts += 1
        self.log(f"  Prop 9.9 no-good cut {self.cut_count} excludes x={x}")

    def _update_incumbent(self, obj, x, y):
        if obj < self.UB-TOL:
            self.UB=float(obj)
            self.incumbent=(list(x),list(y))
            self.log(f"  Incumbent updated: transformed objective={obj:g}, "
                     f"original leader objective={self.problem.original_leader_obj(obj):g}")

    def _implied_fixed_links(self, node):
        """Detect implied fixings for small models, without changing node LP.

        A variable is fixed when its LP minimum and maximum coincide. For large
        networks probing every arc would be expensive, so only explicitly
        branched fixings are used (safe but potentially redundant branches).
        """
        fixed = set(node.fixed_links)
        if self.problem.nx > 8:
            return fixed
        for name in self.problem.linking_names:
            if name in fixed:
                continue
            probe = node.model.copy()
            probe.Params.OutputFlag = 0
            v = probe.getVarByName(name)
            probe.setObjective(v, GRB.MINIMIZE)
            probe.optimize()
            if probe.Status != GRB.OPTIMAL:
                raise RuntimeError("Cannot certify implied variable fixing")
            lo = probe.ObjVal
            probe.setObjective(v, GRB.MAXIMIZE)
            probe.optimize()
            if probe.Status != GRB.OPTIMAL:
                raise RuntimeError("Cannot certify implied variable fixing")
            hi = probe.ObjVal
            if hi - lo <= TOL:
                fixed.add(name)
        return fixed

    def _process(self, node, start):
        self.log(f"\nNode {node.number} depth={node.depth} {node.branch}")
        while True:
            self._check_time(start)
            node.model.optimize()
            self.lp_count += 1
            status=node.model.Status
            if status == GRB.INFEASIBLE:
                node.status="infeasible";return
            if status != GRB.OPTIMAL:
                raise RuntimeError(f"Node {node.number}: LP returned status {status}")
            node.lp_bound=float(node.model.ObjVal)
            node.solution={v.VarName:v.X for v in node.model.getVars()}
            self.log(f"  LP bound {node.lp_bound:.8g}")
            if node.lp_bound >= self.UB-TOL:
                node.status="fathomed_by_bound";return
            # Integer feasibility is checked on ALL problem-declared discrete vars.
            fractionals=[(name,node.solution[name]) for name in self.problem.integer_names
                         if abs(node.solution[name]-round(node.solution[name]))>TOL]
            if fractionals:
                name,value=fractionals[0]
                self._children(node,name,(floor(value),ceil(value)))
                return
            x=[int(round(node.solution[f"x[{i}]"])) for i in range(self.problem.nx)]
            y=[int(round(node.solution[f"y[{i}]"])) for i in range(self.problem.ny)]
            phi,yf=self.problem.follower(x)
            self.log(f"  follower optimum={phi:g}; candidate={self.problem.candidate_follower_value(x,node.solution):g}")
            if self.problem.check_follower_optimal(x,node.solution,phi):
                self._update_incumbent(node.lp_bound,x,y)
                node.status="bilevel_feasible";return
            result=self.problem.refinement(x,phi)
            if result is not None:
                self._update_incumbent(*result)
            # In Algorithm 7 all linking vars fixed => fathom after refinement.
            # This detects explicit branch fixings; implied fixings may be missed,
            # which is safe for the Prop9.7 strategy but may add extra branches.
            fixed_links = self._implied_fixed_links(node)
            if set(self.problem.linking_names).issubset(fixed_links):
                node.status="fathomed_links_fixed";return
            if self.cut_strategy == "prop97":
                self._add_prop97(node)
            elif self.cut_strategy == "hybrid":
                if result is None:
                    self._add_prop99(node,x)
                else:
                    self._add_prop97(node)
            else:  # prop99 with documented linking-variable branching fallback
                if result is None:
                    self._add_prop99(node,x)
                else:
                    remaining=[name for name in self.problem.linking_names
                               if name not in fixed_links]
                    if not remaining:
                        node.status="fathomed_links_fixed";return
                    # Choose a nonfixed-by-constraints variable when possible;
                    # if all are implied fixed, an explicit fixing branch simply
                    # introduces one redundant and one infeasible child.
                    name=remaining[0]
                    self._children(node,name,(0,1),linking=True)
                    return
            if self.cut_count > self.max_cuts:
                raise RuntimeError("Cut limit reached; optimality NOT certified")

    def solve(self):
        self.nodes={};self.stack=[];self.next_number=1
        self.UB=inf;self.incumbent=None;self.cut_count=0;self.lp_count=0
        self.status="RUNNING"
        root=self.problem.root_lp()
        root.Params.OutputFlag=0
        root.Params.Method=1 # dual simplex -> LP basic vertex
        self.nodes[0]=Node(0,root)
        self.stack=[0]
        start=monotonic()
        try:
            while self.stack:
                self._check_time(start)
                if len(self.nodes)>self.max_nodes:
                    raise RuntimeError("Node limit reached; optimality NOT certified")
                self._process(self.nodes[self.stack.pop()],start)
        except (RuntimeError,ValueError,TimeoutError) as ex:
            self.status="INCOMPLETE"
            print(f"INCOMPLETE: {ex}")
            raise
        self.status="OPTIMAL" if self.incumbent is not None else "INFEASIBLE"
        self.report()
        return self.UB,self.incumbent

    def report(self):
        print(f"\n{self.problem.name} | cut mode={self.cut_strategy} | status={self.status}")
        print("Nodes:",len(self.nodes),"LP solves:",self.lp_count,"Cuts:",self.cut_count)
        if self.incumbent is None:
            print("No bilevel-feasible incumbent")
        else:
            print("Optimal value (original leader sense):",
                  self.problem.original_leader_obj(self.UB))
            print("Leader/follower solution:",self.problem.description(*self.incumbent))

    def plot_tree(self, filename="tree.png", show=False):
        children={i:[] for i in self.nodes}
        for i,node in self.nodes.items():
            if node.parent is not None:
                children[node.parent].append(i)
        positions={};leaf_counter=[0]
        def locate(i):
            kids=sorted(children[i])
            if kids:
                xs=[locate(j) for j in kids]
                x=sum(xs)/len(xs)
            else:
                x=leaf_counter[0];leaf_counter[0]+=1
            positions[i]=(x,-self.nodes[i].depth)
            return x
        locate(0)
        height=1+max(n.depth for n in self.nodes.values())
        fig,ax=plt.subplots(figsize=(max(10,2.9*leaf_counter[0]),max(5,1.9*height)))
        for i,node in self.nodes.items():
            x,y=positions[i]
            if node.parent is not None:
                px,py=positions[node.parent]
                ax.plot([px,x],[py,y],color="0.5",zorder=1)
                ax.text((px+x)/2,(py+y)/2+0.11,node.branch,fontsize=8,ha="center")
        for i,node in self.nodes.items():
            x,y=positions[i]
            lines=[f"Node {i}",node.status]
            if node.lp_bound is not None:lines.append(f"bound={node.lp_bound:.4g}")
            if node.cuts:lines.append(f"cuts={node.cuts}")
            ax.text(x,y,"\n".join(lines),ha="center",va="center",fontsize=8,zorder=3,
                    bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="black"))
        ax.set_xlim(-0.8,leaf_counter[0]-0.2)
        ax.set_ylim(-height+0.3,0.7)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(filename,dpi=180,bbox_inches="tight")
        if show:plt.show()
        plt.close(fig)
        print("Saved:",filename)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--problem",choices=("915","path-objective","path-constraint"),default="915")
    parser.add_argument("--cuts",choices=("prop97","hybrid","prop99"),default="prop97")
    parser.add_argument("--max-nodes",type=int,default=10000)
    parser.add_argument("--max-cuts",type=int,default=20000)
    parser.add_argument("--time-limit",type=float,default=None)
    parser.add_argument("--quiet",action="store_true")
    parser.add_argument("--plot",default=None)
    args=parser.parse_args()
    problem=(Knapsack915() if args.problem=="915" else
             PathInterdiction("objective" if args.problem=="path-objective" else "constraint"))
    solver=BranchAndCut(problem,args.cuts,args.max_nodes,args.max_cuts,args.time_limit,
                        verbose=not args.quiet)
    solver.solve()
    if args.plot:solver.plot_tree(args.plot)


if __name__ == "__main__":
    main()
