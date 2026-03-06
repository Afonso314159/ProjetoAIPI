import random
import time
from puzzle_utils import findPossibleMoves, GOALS

def DFS(start, time_limit_seconds, ordem="standard"):
    goal_data = GOALS.get(ordem, GOALS["standard"])
    goal = goal_data["list"]

    stack = [start]
    
    # Dicionário para guardar {estado_filho: estado_pai}
    parent_map = {tuple(map(tuple, start)): None}
    visited = set()
    start_time = time.time()
    nodes_expanded = 0

    while stack:
        if time.time() - start_time > time_limit_seconds:
            print("Tempo limite atingido.")
            return None
        
        matriz_atual = stack.pop()
        atual_tuplo = tuple(map(tuple, matriz_atual))
        
        if atual_tuplo in visited:
            continue
            
        visited.add(atual_tuplo)
        nodes_expanded += 1

        if matriz_atual == goal:
            print(f"Solução Encontrada! Estados expandidos: {nodes_expanded}")
            # Reconstrói o caminho
            path = []
            curr = atual_tuplo
            while curr is not None:
                path.append([list(row) for row in curr])
                curr = parent_map[curr]
            return path[::-1]

        moves = findPossibleMoves(matriz_atual)
        for move in moves:
            move_tuplo = tuple(map(tuple, move))
            if move_tuplo not in visited and move_tuplo not in [tuple(map(tuple, s)) for s in stack]:
                parent_map[move_tuplo] = atual_tuplo
                stack.append(move)

    return None

def resolucao_DFS(matrix, time_limit_seconds, ordem):
    return DFS(matrix, time_limit_seconds, ordem)

