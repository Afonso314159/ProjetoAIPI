import random
BOARD_SIZE = 4

# -------------------------------
# Goal state and positions
# -------------------------------

GOALS = {
    "standard": {
        "list": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]],
        "tuple": ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 0)),
        "dic": {1: (0,0), 2: (0,1), 3: (0,2), 4: (0,3), 5: (1,0), 6: (1,1), 7: (1,2), 8: (1,3),
                9: (2,0), 10: (2,1), 11: (2,2), 12: (2,3), 13: (3,0), 14: (3,1), 15: (3,2), 0: (3,3)}
    },
    "backwards": {
        "list": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        "tuple": ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)),
        "dic": {0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (1,0), 5: (1,1), 6: (1,2), 7: (1,3),
                8: (2,0), 9: (2,1), 10: (2,2), 11: (2,3), 12: (3,0), 13: (3,1), 14: (3,2), 15: (3,3)}
    }
}

# -------------------------------
# Move generation
# -------------------------------

def findPossibleMoves(matrix):
    matrixEmLista = [num for linha in matrix for num in linha]
    espacoVazioIndex = matrixEmLista.index(0)
    linha, coluna = divmod(espacoVazioIndex, 4)
    moves_possiveis = []
    moves_finais = []

    #Restrições 1ª Linha
    if linha > 0 :
        moves_possiveis.append(espacoVazioIndex - 4)

    # Restrições última linha
    if linha < 3:
        moves_possiveis.append(espacoVazioIndex + 4)

    # Restrições 1ª Coluna
    if coluna > 0:
        moves_possiveis.append(espacoVazioIndex - 1)

    # Restrições última coluna
    if coluna < 3:
        moves_possiveis.append(espacoVazioIndex + 1)

    for idx in moves_possiveis:
        novamatrix = matrixEmLista.copy()
        novamatrix[espacoVazioIndex] , novamatrix[idx] = novamatrix[idx] , novamatrix[espacoVazioIndex]
        moves_finais.append([novamatrix[i:i+4] for i in range(0,16,4)])
    return moves_finais


# -------------------------------
# Heuristic
# -------------------------------

def manhattanDistance(flat_state, target_dic):
    dist = 0
    for i, tile in enumerate(flat_state):
        if tile == 0:
            continue
        r, c = divmod(i, 4)
        gr, gc = target_dic[tile]
        dist += abs(r - gr) + abs(c - gc)
    return dist


# -------------------------------
# Path reconstruction
# -------------------------------

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def isSolvable(matrix):
    """
    Verifica se a matriz do puzzle 15 é solucionável.
    Para um puzzle 4x4:
    - Conta o número de inversões (pares onde um número maior aparece antes de um menor)
    - Encontra a linha do espaço vazio (0) contando de baixo para cima
    - É solucionável se: (inversões é par e linha do 0 é ímpar) OU (inversões é ímpar e linha do 0 é par)
    """
    # Achatar a matriz em uma lista única
    flat = [num for row in matrix for num in row]
    
    # Contar inversões (ignorando o 0)
    inversions = 0
    for i in range(len(flat)):
        if flat[i] == 0:
            continue
        for j in range(i + 1, len(flat)):
            if flat[j] == 0:
                continue
              
            if flat[i] > flat[j]:
                    inversions += 1

    # Encontrar a linha do 0 (contando de baixo para cima, começando em 1)
    zero_row_from_bottom = 0
    for i, row in enumerate(matrix):
        if 0 in row:
            zero_row_from_bottom = 4 - i  # 4 linhas total, contando de baixo
            break
    
    # Para puzzle 4x4:
    # Se a linha do 0 (de baixo) é ímpar, inversões deve ser par
    # Se a linha do 0 (de baixo) é par, inversões deve ser ímpar
    if zero_row_from_bottom % 2 == 1:  # linha ímpar
        return inversions % 2 == 0
    else:  # linha par
        return inversions % 2 == 1


def generateMatrix(ordem="standard"):
    """
    Gera uma matriz aleatória 4x4 para o puzzle dos 15.
    """
    while True:
        matrix = list(range(16))
        random.shuffle(matrix)
        
        # Organizar em matriz 4x4
        arrangedMatrix = [matrix[i:i+4] for i in range(0, len(matrix), 4)]
        
        return arrangedMatrix
    
def manhattan_linear_conflict(state, target_dic):
    manhattan = 0
    linear_conflict = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            tile = state[r][c]
            if tile == 0:
                continue
            tr, tc = target_dic[tile]
            manhattan += abs(r - tr) + abs(c - tc)

    # Row conflicts
    for r in range(BOARD_SIZE):
        tiles = []
        for c in range(BOARD_SIZE):
            tile = state[r][c]
            if tile != 0 and target_dic[tile][0] == r:
                tiles.append(tile)
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if target_dic[tiles[i]][1] > target_dic[tiles[j]][1]:
                    linear_conflict += 2

    # Column conflicts
    for c in range(BOARD_SIZE):
        tiles = []
        for r in range(BOARD_SIZE):
            tile = state[r][c]
            if tile != 0 and target_dic[tile][1] == c:
                tiles.append(tile)
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if target_dic[tiles[i]][0] > target_dic[tiles[j]][0]:
                    linear_conflict += 2

    return manhattan + linear_conflict