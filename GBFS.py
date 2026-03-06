import random
from heapq import heappush, heappop
from puzzle_utils import  GOALS, findPossibleMoves, manhattanDistance, reconstruct_path
import time


def gbfs(start_matrix, time_limit_seconds, ordem="standard"):
    
    target_data = GOALS.get(ordem, GOALS["standard"])
    target_tuple = target_data["tuple"]
    target_dic = target_data["dic"]

    pq = []
    start_tuple = tuple(tuple(l) for l in start_matrix)
    
    h0 = manhattanDistance([num for lin in start_matrix for num in lin], target_dic)
    heappush(pq, (h0, start_tuple))

    start_time = time.time()
    anterior = {}
    visited = {start_tuple}
    expansions = 0

    while pq:
        if time.time() - start_time > time_limit_seconds:
            print("Tempo limite atingido.")
            return None
        
        h, estadoAtual = heappop(pq)
        expansions += 1
        
        if estadoAtual == target_tuple:
            tempo = time.time() - start_time
            path = reconstruct_path(anterior, estadoAtual)
            print(f"Solução encontrada em {len(path) - 1} movimentos\n")
            print(f"Tempo: {tempo}")
            print(f"Estados expandidos: {expansions}")
            return path

        for nxt in findPossibleMoves(estadoAtual):
            nxt_tuple = tuple(tuple(l) for l in nxt)
            if nxt_tuple not in visited:
                anterior[nxt_tuple] = estadoAtual
                visited.add(nxt_tuple)
                h_nxt = manhattanDistance([num for lin in nxt for num in lin], target_dic)
                heappush(pq, (h_nxt, nxt_tuple))
                
    print("Sem solução.")
    return None

def resolucao_GBFS(matrix, time_limit_seconds, ordem):
    return gbfs(matrix, time_limit_seconds, ordem)