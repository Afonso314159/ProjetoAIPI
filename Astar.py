import random
from heapq import heappush, heappop
from puzzle_utils import GOALS, findPossibleMoves, manhattanDistance, reconstruct_path
import time


def astar(start_matrix, time_limit_seconds, ordem="standard"):

    goal_data = GOALS.get(ordem, GOALS["standard"])
    target_tuple = goal_data["tuple"]
    target_dic = goal_data["dic"]

    start_tuple = tuple(tuple(r) for r in start_matrix)
    pq = []
    h0 = manhattanDistance([n for row in start_matrix for n in row], target_dic)
    heappush(pq, (h0, 0, start_tuple))

    start_time = time.time()
    came_from = {}
    visited = set()
    expansions = 0

    g_score = {start_tuple: 0}

    while pq:
        f, g, current = heappop(pq)

        if time.time() - start_time > time_limit_seconds:
            return None

        if current in visited:
            continue

        visited.add(current)

        if current == target_tuple:
            tempo = time.time() - start_time
            path = reconstruct_path(came_from, current)
            print(f"Solução encontrada em {len(path) - 1} movimentos\n")
            print(f"Tempo: {tempo}")
            print(f"Estados expandidos: {expansions}")
            return path

        current_matrix = [list(r) for r in current]

        expansions += 1
        for nxt in findPossibleMoves(current_matrix):
            nxt_tuple = tuple(tuple(r) for r in nxt)

            tentative_g = g + 1
            if tentative_g < g_score.get(nxt_tuple, float("inf")):
                g_score[nxt_tuple] = tentative_g
                came_from[nxt_tuple] = current
                h = manhattanDistance([n for row in nxt for n in row], target_dic)
                heappush(pq, (tentative_g + h, tentative_g, nxt_tuple))

            
    return None

def resolucao_Astar(matrix, time_limit_seconds, ordem):
    return astar(matrix, time_limit_seconds, ordem)