import cv2
import numpy as np
import matplotlib.pyplot as plt
from cnn import get_model, predict_number_with_confidence

#G4
def processamento_da_imagem(file_name, identification_method="template"):
    img_rgb = plt.imread(file_name)
    
    #Preparar Imagem
    escala = 1200 / max(img_rgb.shape[:2])
    img_rgb = cv2.resize(img_rgb, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Segmentar vermelho
    mask_r1 = cv2.inRange(img_hsv, (0, 80, 60), (10, 255, 255))
    mask_r2 = cv2.inRange(img_hsv, (170, 80, 60), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    # Segmentar branco
    mask_white = cv2.inRange(img_hsv, (0, 0, 160), (180, 80, 255))

    mask_red = cv2.medianBlur(mask_red, 5)
    mask_white = cv2.medianBlur(mask_white, 5)

    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

    mask_r_d = cv2.dilate(mask_red, ker, iterations = 1)
    mask_w_d = cv2.dilate(mask_white, ker, iterations = 1)

    inter = cv2.bitwise_and(mask_r_d, mask_w_d)

    num_r, labels_r = cv2.connectedComponents(mask_r_d, connectivity = 8)

    num_w, labels_w = cv2.connectedComponents(mask_w_d)
    
    ids_r = np.unique(labels_r[inter > 0])
    ids_w = np.unique(labels_w[inter > 0])
    
    # Remover fundo (0)
    ids_r = ids_r[ids_r != 0]
    ids_w = ids_w[ids_w != 0]

    # Máscaras dos componentes que tocam
    mask_r_touch = np.zeros_like(labels_r, dtype=np.uint8)
    for rid in ids_r:
        mask_r_touch |= (labels_r == rid)
    mask_r_touch = (mask_r_touch * 255).astype(np.uint8)
    
    mask_w_touch = np.zeros_like(labels_w, dtype=np.uint8)
    for wid in ids_w:
        mask_w_touch |= (labels_w == wid)
    mask_w_touch = (mask_w_touch * 255).astype(np.uint8)
    
    # União das duas regiões (Vermelho + Branco)
    mask_tiles_touch = cv2.bitwise_or(mask_r_touch, mask_w_touch)
    mask = cv2.erode(mask_tiles_touch, ker, iterations=1)
    
    # Obter contornos e polígono final
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Nenhum contorno encontrado.")
    
    largest = max(contours, key=cv2.contourArea)
    perimetro = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.01 * perimetro, True)
    
    mask_refinada = np.zeros_like(mask)
    cv2.drawContours(mask_refinada, [approx], -1, 255, thickness=-1)

    contours, hierarchy = cv2.findContours(mask_refinada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # Obter polígono
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2)

    # Calcular todos os segmentos de reta entre vértices consecutivos do polígono
    segments = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        length = np.linalg.norm(p1 - p2)
        segments.append((length, p1, p2))

    # Deixar apenas os 4 maiores segmentos de reta
    segments.sort(key=lambda x: x[0], reverse=True)
    top4 = segments[:4]

    # Função para encontrar a interseção de duas retas
    def line_intersection(p1, p2, p3, p4):
        # Calcula o ponto de interseção das retas definidas por (p1,p2) e (p3,p4)
        # Retorna None se as retas forem paralelas
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if abs(denom) < 1e-6:
            return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return np.array([px, py])

    # Calcular as interseções entre todas as combinações das quatro maiores retas
    intersections = []
    for i in range(len(top4)):
        for j in range(i+1, len(top4)):
            inter = line_intersection(top4[i][1], top4[i][2], top4[j][1], top4[j][2])
            if inter is not None:
                intersections.append(inter)

    # Calcular o centróide do polígono
    M = cv2.moments(approx)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = np.array([cx, cy])

    # Selecionar os 4 cantos corretos
    if len(intersections) > 4:
        intersections = sorted(intersections, key=lambda p: np.linalg.norm(p - centroid))[:4]

    overlay = img_rgb.copy()
    for pt in intersections:
        cv2.circle(overlay, tuple(np.int32(pt)), 6, (0,255,0), -1)
    cv2.circle(overlay, (cx, cy), 6, (255,0,0), -1)  # centroid
    
    for (_, p1, p2) in top4:
        cv2.line(overlay, tuple(p1), tuple(p2), (0,0,255), 2)
    
    alpha = 0.6
    output = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)

    # Ordenar os 4 cantos
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    # Converter lista de interseções para matriz
    corners = np.array(intersections, dtype="float32")

    # Ordenar pontos do polígono (TL,TR,BR,BL)
    rect = order_points(corners)

    mask_quad_final = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    quad_pts = rect.astype(np.int32).reshape((-1, 1, 2))
    cv2.drawContours(mask_quad_final, [quad_pts], -1, 255, thickness=-1)

    # Transformação para perspetiva
    dst = np.array([
        [100, 100],
        [600, 100],
        [600, 600],
        [100, 600]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(output, M, (700, 700))
    board = warped[100:600, 100:600]

    board_hsv = cv2.cvtColor(board, cv2.COLOR_RGB2HSV)
    H = board_hsv[:,:,0]

    lower_gold = (9, 10, 10)
    median_hue = np.median(H)
    if median_hue <= 14:
        lower_gold = (6, 0, 0)

    if median_hue > 100:
        lower_gold = (3, 0, 0)
        
    upper_gold = (35, 255, 255) 
    mask_gold = cv2.inRange(board_hsv, lower_gold, upper_gold)

    kerneldlt = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_gold_dlt = cv2.dilate(mask_gold, kerneldlt, iterations = 1)

    mask_gold_clean = cv2.medianBlur(mask_gold_dlt, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    mask_gold_clean = cv2.morphologyEx(mask_gold_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    templates = {}
    cnn_model = None

    if identification_method == "template":
        for k in range(1, 16):
            t = cv2.imread(f"15GameMasks/mask{k}.png", cv2.IMREAD_GRAYSCALE)
            if t is None:
                print(f"AVISO: Template {k} não encontrado!")
                continue
            templates[k] = t
    else:
        cnn_model = get_model()

    board_visualization = board.copy()

    detected_board = np.zeros((4, 4), dtype=int)

    # Processar cada célula da matriz 4x4
    cell_h = board.shape[0] // 4
    cell_w = board.shape[1] // 4

    empty_tile_position = None
    numbers_scores = list()

    for row in range(4):
        for col in range(4):
            # Definir região da célula
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            
            # Extrair máscara da célula
            cell_mask = mask_gold_clean[y1:y2, x1:x2]
            
            # Encontrar contornos na célula e filtra-los contornos por área mínima
            contours_cell, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours_cell if cv2.contourArea(c) > 150]
            
            if valid_contours:
                # Há número nesta célula - fazer template matching
                
                # Calcular bounding box que engloba TODOS os contornos
                all_points = []
                for contour in valid_contours:
                    all_points.extend(contour.reshape(-1, 2))
                
                all_points = np.array(all_points)
                
                x_min = np.min(all_points[:, 0])
                y_min = np.min(all_points[:, 1])
                x_max = np.max(all_points[:, 0])
                y_max = np.max(all_points[:, 1])
                
                # Extrair região do número (com pequena margem)
                margin = 5
                x_min_m = max(0, x_min - margin)
                y_min_m = max(0, y_min - margin)
                x_max_m = min(cell_mask.shape[1], x_max + margin)
                y_max_m = min(cell_mask.shape[0], y_max + margin)
                
                number_region = cell_mask[y_min_m:y_max_m, x_min_m:x_max_m]
                                
                if identification_method == "template":
                    # Template Matching
                    best_match_score = -1
                    best_match_num = 0
                    
                    for num, template in templates.items():
                        # Ajustar tamanho do template para o tamanho da região
                        if number_region.shape[0] > 0 and number_region.shape[1] > 0:
                            template_resized = cv2.resize(template, 
                                                        (number_region.shape[1], number_region.shape[0]),
                                                        interpolation=cv2.INTER_AREA)
                            
                            # Usar TM_CCOEFF_NORMED (valores entre -1 e 1, quanto maior melhor)
                            result = cv2.matchTemplate(number_region, template_resized, cv2.TM_CCOEFF_NORMED)
                            score = result[0, 0]
                            
                            if score > best_match_score:
                                best_match_score = score
                                best_match_num = num

                    detected_num = best_match_num
                    detected_score = float(best_match_score)
                else:
                    detected_num, detected_score = predict_number_with_confidence(
                        number_region,
                        cnn_model,
                    )
                
                # Guardar na matriz
                detected_board[row, col] = detected_num
                numbers_scores.append([detected_num, float(detected_score), row, col])
                
                # Calcular centroide
                cx = (x_min + x_max) // 2 + x1
                cy = (y_min + y_max) // 2 + y1
                
                # Ajustar coordenadas para o board completo
                x_abs = x1 + x_min
                y_abs = y1 + y_min
                w_abs = x_max - x_min
                h_abs = y_max - y_min
                
                # Desenhar retângulo ao redor do número completo
                cv2.rectangle(board_visualization,
                            (x_abs, y_abs),
                            (x_abs + w_abs, y_abs + h_abs),
                            (0, 255, 0), 2)
                
                # Adicionar número detectado
                cv2.putText(board_visualization, f"{detected_num}",
                        (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Adicionar score (para debug)
                cv2.putText(board_visualization, f"{detected_score:.2f}",
                        (x_abs, y_abs - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            else:
                # Esta é a célula vazia
                detected_board[row, col] = 0  # 0 representa espaço vazio
                numbers_scores.append([0, -999.0, row, col])  # Adicionar também à lista
                
                # Calcular centro da célula vazia
                cx_empty = x1 + cell_w // 2
                cy_empty = y1 + cell_h // 2

                # Adicionar texto "0"
                cv2.putText(board_visualization, "0",
                        (cx_empty - 10, cy_empty + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Verificar números repetidos e encontrar espaço vazio
    seen = set()
    duplicates = set()

    # Identificar quais números estão duplicados (excluindo 0)
    for num_scores in numbers_scores:
        if num_scores[0] == 0:  # Ignorar o 0
            continue
        if num_scores[0] in seen:
            duplicates.add(num_scores[0])
        else:
            seen.add(num_scores[0])

    empty_tile_position = None

    if duplicates:
        print(f"Números duplicados encontrados: {duplicates}")
        
        for dup_num in duplicates:
            # Filtrar apenas as ocorrências deste número
            occurrences = [info for info in numbers_scores if info[0] == dup_num]
            
            # Encontrar a ocorrência com menor score
            min_occurrence = min(occurrences, key=lambda x: x[1])
            empty_num, empty_score, empty_row, empty_col = min_occurrence
            
            print(f"Espaço vazio identificado em ({empty_row},{empty_col}) com score {empty_score:.4f}")
            
            # Marcar como espaço vazio
            detected_board[empty_row, empty_col] = 0
            empty_tile_position = (empty_row, empty_col)
    else:
        # Se não houver duplicados, procurar o 0 na lista
        for info in numbers_scores:
            if info[0] == 0:
                empty_tile_position = (info[2], info[3])
                break
    
    
    # Corrige multiplos 0s (a CNN está defaulting muitos 0s)
    zeros_in_board = [(info[2], info[3], info[1]) for info in numbers_scores if info[0] == 0]
    
    if len(zeros_in_board) > 1:
        print(f"Múltiplos zeros detectados ({len(zeros_in_board)}).")
        
        # A verdadeira célula vazia é a célula com score -999
        # Se nenhum tiver score de -999, usar a célula com menor score
        true_empty_candidates = [z for z in zeros_in_board if z[2] == -999.0]
        
        if true_empty_candidates:
            true_empty = true_empty_candidates[0]
        else:
            true_empty = min(zeros_in_board, key=lambda x: x[2])
        
        empty_tile_position = (true_empty[0], true_empty[1])
        
        # Encontrar quais números não estão presentes na board (1-15 + 0)
        flat = detected_board.flatten().tolist()
        present_numbers = set(flat) - {0}
        # Numeros que deveriam estar presentes mas não estão
        missing_numbers = set(range(1, 16)) - present_numbers  
        
        # Encontrar as células que foram classificadas como 0 erradamente
        false_zeros = [(z[0], z[1], z[2]) for z in zeros_in_board if (z[0], z[1]) != empty_tile_position]
        
        # Ordenar os zeros falsos por nível de confiança
        # (maior confiança = maior probabilidade de ser um dígito real não detetado)
        # e atribuir-lhes os números em falta
        false_zeros_sorted = sorted(false_zeros, key=lambda x: x[2], reverse=True)
        missing_numbers_list = sorted(missing_numbers)

        for i, (row, col, score) in enumerate(false_zeros_sorted):
            if i < len(missing_numbers_list):
                detected_board[row, col] = missing_numbers_list[i]
                print(f"  Peça ({row},{col}) reatribuída de 0 para {missing_numbers_list[i]}")
            else:
                # Caso existam mais zeros falsos do que números em falta, manter o valor atual
                # (situação que não deverá ocorrer)
                pass


    print(f"\nEspaço vazio encontrado na posição: {empty_tile_position}")
    print("\nTabuleiro detetado (0 = vazio):")
    print(detected_board)

    # Mostrar resultados
    plt.figure(figsize=(20,10))

    plt.subplot(1, 6, 1)
    plt.imshow(img_rgb)
    plt.title("Board Original")
    plt.axis("off")

    plt.subplot(1, 6, 2)
    plt.imshow(output)
    plt.title("Cantos Identificados")
    plt.axis("off")

    plt.subplot(1, 6, 3)
    plt.imshow(mask_quad_final,cmap="grey")
    plt.title("Quadrilátero do 15-puzzle identificado")
    plt.axis("off")

    plt.subplot(1, 6, 4)
    plt.imshow(board)
    plt.title("Warp")
    plt.axis("off")

    plt.subplot(1, 6, 5)
    plt.imshow(mask_gold_clean, cmap='gray')
    plt.title("Máscara Gold Limpa")
    plt.axis("off")

    plt.subplot(1, 6, 6)
    plt.imshow(board_visualization)
    title = "Números Detectados\nvia Template Matching"
    if identification_method == "cnn":
        title = "Números Detectados via CNN"
    plt.title(title)
    plt.axis("off")

    plt.show()

    flat = [num for row in detected_board for num in row]

    # Verificar se a matriz contém os números corretos (0-15)
    if set(flat) == set(range(16)):
        return detected_board
    else:
        return None

#processamento_da_imagem(file_name,"cnn")