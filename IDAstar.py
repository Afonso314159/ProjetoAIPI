import time
from puzzle_utils import findPossibleMoves, GOALS, manhattan_linear_conflict

def ida_star(matrix, time_limit, ordem="standard"):
    goal_data = GOALS[ordem]
    target = goal_data["tuple"]
    target_dic = goal_data["dic"]

    start = tuple(tuple(r) for r in matrix)
    start_time = time.time()

    # Heuristic cache
    h_cache = {}

    def heuristic(state):
        if state in h_cache:
            return h_cache[state]
        h = manhattan_linear_conflict([list(r) for r in state], target_dic)
        h_cache[state] = h
        return h

    threshold = heuristic(start)
    path = [start]

    expansions = 0

    def dfs(state, parent, g, threshold):
        nonlocal expansions
        if time.time() - start_time > time_limit:
            return None, float("inf")

        f = g + heuristic(state)
        if f > threshold:
            return None, f
        
        expansions += 1

        if state == target:
            tempo = time.time() - start_time
            print(f"Solução encontrada em {len(path) - 1} movimentos\n")
            print(f"Tempo: {tempo}")
            print(f"Estados expandidos: {expansions}")
            return list(path), f

        min_excess = float("inf")

        # Generate and order successors
        successors = []
        for nxt in findPossibleMoves([list(r) for r in state]):
            nxt_t = tuple(tuple(r) for r in nxt)
            if nxt_t == parent:
                continue
            successors.append((heuristic(nxt_t), nxt_t))

        successors.sort(key=lambda x: x[0])

        for _, nxt_t in successors:
            path.append(nxt_t)
            result, excess = dfs(nxt_t, state, g + 1, threshold)
            if result is not None:
                return result, excess

            min_excess = min(min_excess, excess)
            path.pop()

        return None, min_excess

    while True:
        result, new_threshold = dfs(start, None, 0, threshold)

        if result is not None:
            return result

        if new_threshold == float("inf"):
            return None

        threshold = new_threshold


def resolucao_IDAstar(matrix, time_limit, ordem):
    return ida_star(matrix, time_limit, ordem)
