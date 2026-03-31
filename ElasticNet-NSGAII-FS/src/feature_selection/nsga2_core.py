from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from src.feature_selection.objectives import objectives
from src.feature_selection.operators import guided_init_population, one_point_crossover, guided_mutation

@dataclass
class Individual:
    s: np.ndarray                 # chromosome (d,)
    f: Tuple[float, float, float] # objectives
    rank: int = 0
    cd: float = 0.0

def dominates(a: Tuple[float,...], b: Tuple[float,...]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def fast_non_dominated_sort(pop: List[Individual]) -> List[List[int]]:
    S = [set() for _ in pop]
    n = [0 for _ in pop]
    fronts: List[List[int]] = [[]]

    for p_i, p in enumerate(pop):
        for q_i, q in enumerate(pop):
            if p_i == q_i:
                continue
            if dominates(p.f, q.f):
                S[p_i].add(q_i)
            elif dominates(q.f, p.f):
                n[p_i] += 1
        if n[p_i] == 0:
            p.rank = 0
            fronts[0].append(p_i)

    i = 0
    while fronts[i]:
        next_front = []
        for p_i in fronts[i]:
            for q_i in S[p_i]:
                n[q_i] -= 1
                if n[q_i] == 0:
                    pop[q_i].rank = i + 1
                    next_front.append(q_i)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # last empty
    return fronts

def crowding_distance(pop: List[Individual], front: List[int]) -> None:
    if not front:
        return
    m = len(pop[0].f)
    for idx in front:
        pop[idx].cd = 0.0
    for k in range(m):
        front_sorted = sorted(front, key=lambda i: pop[i].f[k])
        pop[front_sorted[0]].cd = float("inf")
        pop[front_sorted[-1]].cd = float("inf")
        fmin = pop[front_sorted[0]].f[k]
        fmax = pop[front_sorted[-1]].f[k]
        denom = (fmax - fmin) if (fmax - fmin) != 0 else 1.0
        for j in range(1, len(front_sorted) - 1):
            prev_f = pop[front_sorted[j - 1]].f[k]
            next_f = pop[front_sorted[j + 1]].f[k]
            pop[front_sorted[j]].cd += (next_f - prev_f) / denom

def tournament_select(pop: List[Individual], rng: np.random.Generator) -> Individual:
    a = pop[int(rng.integers(0, len(pop)))]
    b = pop[int(rng.integers(0, len(pop)))]
    # lower rank better; if tie, higher crowding distance better
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.cd >= b.cd else b

def nsga2_guided(X: np.ndarray, y: np.ndarray, w: np.ndarray, cfg: dict) -> List[Individual]:
    ns = cfg["nsga2"]
    N = int(ns["population_size"])
    G = int(ns["generations"])
    pc = float(ns["crossover_rate"])
    p_base = float(ns["base_mutation_rate"])
    gamma = float(ns["guidance_gamma"])

    rng = np.random.default_rng(int(cfg["seed"]))

    # init
    Pbin = guided_init_population(N, X.shape[1], w, gamma, rng)
    pop = [Individual(s=Pbin[i], f=objectives(X, y, Pbin[i], cfg)) for i in range(N)]

    fronts = fast_non_dominated_sort(pop)
    for fr in fronts:
        crowding_distance(pop, fr)

    for _ in range(G):
        # mating
        offspring = []
        while len(offspring) < N:
            p1 = tournament_select(pop, rng).s
            p2 = tournament_select(pop, rng).s
            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < pc:
                c1, c2 = one_point_crossover(p1, p2, rng)
            c1 = guided_mutation(c1, w, p_base, rng)
            c2 = guided_mutation(c2, w, p_base, rng)
            offspring.append(Individual(s=c1, f=objectives(X, y, c1, cfg)))
            if len(offspring) < N:
                offspring.append(Individual(s=c2, f=objectives(X, y, c2, cfg)))

        R = pop + offspring
        fronts = fast_non_dominated_sort(R)
        for fr in fronts:
            crowding_distance(R, fr)

        # environmental selection
        new_pop: List[Individual] = []
        for fr in fronts:
            if len(new_pop) + len(fr) <= N:
                new_pop.extend([R[i] for i in fr])
            else:
                # sort by crowding distance desc
                fr_sorted = sorted(fr, key=lambda i: R[i].cd, reverse=True)
                need = N - len(new_pop)
                new_pop.extend([R[i] for i in fr_sorted[:need]])
                break
        pop = new_pop

    # final non-dominated set
    fronts = fast_non_dominated_sort(pop)
    nd = [pop[i] for i in fronts[0]] if fronts else pop
    return nd
