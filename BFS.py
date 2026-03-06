import random
import time
from puzzle_utils import GOALS, findPossibleMoves

def BFS(start_matrix, time_limit_seconds, ordem="standard"):
    goal_data = GOALS.get(ordem, GOALS["standard"])
    goal = goal_data["list"]
    
    queue = [start_matrix]
    
    # Converte matriz inicial para tuplo para ser usado como chave
    start_tuple = tuple(map(tuple, start_matrix))
    
    # parent_map guarda {estado_filho: estado_pai} para reconstruir o caminho
    parent_map = {start_tuple: None}
    visited = {start_tuple}
    
    start_time = time.time()
    nodes_expanded = 0

    while queue:
        if time.time() - start_time > time_limit_seconds:
            print("Tempo limite atingido.")
            return None
        
        matriz_atual = queue.pop(0)
        nodes_expanded += 1

        if matriz_atual == goal:
            tempo = time.time() - start_time
            print(f"Solução Encontrada em {tempo:.2f}s")
            print(f"Estados expandidos: {nodes_expanded}")
            
            # Reconstrói o caminho de trás para a frente
            path = []
            curr = tuple(map(tuple, matriz_atual))
            while curr is not None:
                path.append([list(row) for row in curr])
                curr = parent_map[curr]
            
            return path[::-1]

        for nxt in findPossibleMoves(matriz_atual):
            nxt_tuple = tuple(map(tuple, nxt))
            if nxt_tuple not in visited:
                visited.add(nxt_tuple)
                parent_map[nxt_tuple] = tuple(map(tuple, matriz_atual))
                queue.append(nxt)

    print("Nenhuma solução encontrada.")
    return None

def resolucao_BFS(matrix, time_limit_seconds, ordem):
    return BFS(matrix, time_limit_seconds, ordem)
