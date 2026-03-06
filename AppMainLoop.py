from tkinter import *
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk # type: ignore
from pi import processamento_da_imagem
from Astar import resolucao_Astar
from BFS import resolucao_BFS
from DFS import resolucao_DFS
from GBFS import resolucao_GBFS
from IDAstar import resolucao_IDAstar
from RL import resolucao_RL
from puzzle_utils import generateMatrix, isSolvable

# Variáveis globais
imagem_path = None
imagem_original = None
imagem_display = None
matriz_atual = None
res_passos = None
modo_atual = None  # 'imagem' ou 'aleatorio'

def selecionar_imagem():
    """Abre dialog para selecionar imagem"""
    global imagem_path, imagem_original, imagem_display, modo_atual
    
    file_path = filedialog.askopenfilename(
        title="Selecione a imagem do jogo dos 15",
        filetypes=[("Imagens", "*.png *.jpg *.jpeg *.gif *.bmp"), ("Todos os arquivos", "*.*")]
    )
    
    if file_path:
        try:
            imagem_path = file_path
            imagem_original = Image.open(file_path)
            modo_atual = 'imagem'
            mostrar_imagem()
            
            # Ativar botões
            btn_detetar.config(state=NORMAL)
            btn_resolver.config(state=DISABLED)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar a imagem:\n{str(e)}")

def mostrar_imagem():
    """Mostra a imagem selecionada no frame, garantindo que o widget exista"""
    global imagem_display, label_imagem
    
    if imagem_original:
        # Limpa o frame antes de mostrar nova imagem
        for widget in frame_imagem.winfo_children():
            widget.destroy()
            
        # Recria o label_imagem pois ele pode ter sido removido pela geração de matriz
        label_imagem = Label(frame_imagem, bg="white")
        label_imagem.place(relx=0.5, rely=0.5, anchor=CENTER)

        # Redimensionar
        img_copy = imagem_original.copy()
        img_copy.thumbnail((400, 300), Image.Resampling.LANCZOS)
        imagem_display = ImageTk.PhotoImage(img_copy)
        
        label_imagem.config(image=imagem_display)
        label_imagem.image = imagem_display

def detetar_disposicao():
    """Função para detetar disposição dos números"""
    global matriz_atual
    if imagem_path:
        abrir_popup_metodo_identificacao()
    else:
        messagebox.showwarning("Aviso", "Nenhuma imagem selecionada!")


def abrir_popup_metodo_identificacao():
    """Abrir popup para selecionar método de identificação"""
    popup = Toplevel(janela)
    popup.title("Selecionar Método de Identificação")
    popup.geometry("400x220")
    popup.resizable(False, False)
    popup.grab_set()  # Modal
    
    Label(popup, text="Escolha o Método de Identificação", 
          font="nunito 11 bold").pack(pady=20)
    
    # Frame para método
    frame_metodo = Frame(popup)
    frame_metodo.pack(pady=10, padx=20, fill=X)
    
    Label(frame_metodo, text="Método:", font="nunito 9").pack(anchor=W, pady=5)
    
    metodo_var = StringVar(value="Template Matching")
    metodos = ["Template Matching", "CNN"]
    
    combo_metodo = ttk.Combobox(frame_metodo, textvariable=metodo_var, 
                                values=metodos, state="readonly",
                                font="nunito 9", width=25)
    combo_metodo.pack(fill=X)
    
    # Frame para botões
    frame_btns = Frame(popup)
    frame_btns.pack(pady=20)
    
    def executar_detecao():
        """Executa a deteção com o método selecionado"""
        global matriz_atual
        
        metodo = metodo_var.get()
        
        if metodo == "Template Matching":
            identification_method = "template"
        else:
            identification_method = "cnn"
        
        popup.destroy()
        
        try:
            matriz_atual = processamento_da_imagem(imagem_path, identification_method)
            matriz_atual = matriz_atual.tolist()
            
            if matriz_atual is not None:
                matriz_str = str(matriz_atual)
                messagebox.showinfo("Detetar Disposição",
                              f"Método: {metodo}\n\n{matriz_str}\n\nImagem: {imagem_path}")
                btn_resolver.config(state=NORMAL)
            else:
                messagebox.showwarning("Aviso", "Não foi possível detetar os números")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar imagem:\n{str(e)}")
    
    def cancelar():
        """Fecha o popup sem fazer nada"""
        popup.destroy()
    
    Button(frame_btns, text="Detetar", command=executar_detecao,
           bg="#4CAF50", fg="white", font="nunito 9 bold",
           padx=20, pady=5, cursor="hand2").grid(row=0, column=0, padx=5)
    
    Button(frame_btns, text="Cancelar", command=cancelar,
           bg="#f44336", fg="white", font="nunito 9",
           padx=20, pady=5, cursor="hand2").grid(row=0, column=1, padx=5)

def gerar_matriz_aleatoria():
    """Gera matriz baseada na ordem selecionada no menu principal"""
    global matriz_atual, modo_atual
    try:
        
        matriz_atual = generateMatrix()
        modo_atual = 'aleatorio'
        
        mostrar_matriz_no_frame(matriz_atual)
        
        matriz_str = '\n'.join([str(row) for row in matriz_atual])
        messagebox.showinfo("Matriz Gerada",
                          f"Matriz configurada para o modo: {modo_atual}\n\n{matriz_str}")
        
        btn_resolver.config(state=NORMAL)
        btn_detetar.config(state=DISABLED)
        
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao gerar matriz:\n{str(e)}")

def gerar_matriz_aleatoria():
    """Gera matriz baseada na ordem selecionada no menu principal"""
    global matriz_atual, modo_atual
    try:
        
        matriz_atual = generateMatrix()
        modo_atual = 'aleatorio'
        
        mostrar_matriz_no_frame(matriz_atual)
        
        matriz_str = '\n'.join([str(row) for row in matriz_atual])
        messagebox.showinfo("Matriz Gerada",
                          f"Matriz configurada para o modo: {modo_atual}\n\n{matriz_str}")
        
        btn_resolver.config(state=NORMAL)
        btn_detetar.config(state=DISABLED)
        
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao gerar matriz:\n{str(e)}")

def mostrar_matriz_no_frame(matriz):
    """Mostra a matriz gerada visualmente no frame"""
    
    for widget in frame_imagem.winfo_children():
        widget.destroy()
    
    matriz_frame = Frame(frame_imagem, bg="white")
    matriz_frame.place(relx=0.5, rely=0.5, anchor=CENTER)
    
    # Título
    Label(matriz_frame, text="Puzzle Gerado:", font="nunito 11 bold", 
          bg="white").grid(row=0, column=0, columnspan=4, pady=10)
    
    # Mostrar a matriz em formato de grid
    for r, row in enumerate(matriz):
        for c, val in enumerate(row):
            # 0 representa o espaço vazio
            if val == 0:
                bg_color = "#e0e0e0"
                text = ""
            else:
                bg_color = "#4CAF50"
                text = str(val)
            
            Label(matriz_frame, text=text, width=6, height=3, 
                  borderwidth=2, relief="solid", font="nunito 14 bold",
                  bg=bg_color, fg="white").grid(row=r+1, column=c, padx=2, pady=2)

def abrir_popup_algoritmo():
    """Abre popup para selecionar algoritmo e profundidade"""
    
    popup = Toplevel(janela)
    popup.title("Configurar Resolução")
    popup.geometry("450x300")
    popup.resizable(False, False)
    popup.grab_set()  # Modal
    
    Label(popup, text="Selecione o Algoritmo e Profundidade", font="nunito 11 bold").pack(pady=15)
    
    # Frame algoritmo
    frame_algo = Frame(popup)
    frame_algo.pack(pady=10, padx=20, fill=X)
    Label(frame_algo, text="Algoritmo:", font="nunito 9").pack(anchor=W, pady=5)
    algoritmo_var = StringVar(value="IDA*")
    algoritmos = ["IDA*", "A*", "GBFS", "DFS", "BFS", "Reinforcement Learning"]
    
    combo_algoritmo = ttk.Combobox(frame_algo, textvariable=algoritmo_var, 
                                   values=algoritmos, state="readonly",
                                   font="nunito 9", width=25)
    combo_algoritmo.pack(fill=X)
    
    # Frame para profundidade
    frame_time_limit_seconds = Frame(popup)
    frame_time_limit_seconds.pack(pady=10, padx=20, fill=X)
    
    Label(frame_time_limit_seconds, text="Tempo máximo em segundos:",
          font="nunito 9").pack(anchor=W, pady=5)
    
    time_limit_seconds_var = StringVar(value="30")
    entry_time_limit_seconds = Entry(frame_time_limit_seconds, textvariable=time_limit_seconds_var, 
                       font="nunito 9", width=28)
    entry_time_limit_seconds.pack(fill=X)
    
    # Frame para botões
    frame_btns = Frame(popup)
    frame_btns.pack(pady=20)
    
    def executar_resolucao():
        """Executa a resolução com os parâmetros selecionados"""
        global matriz_atual
        
        try:
            algoritmo = algoritmo_var.get()
            time_limit_seconds = int(time_limit_seconds_var.get())
            
            if time_limit_seconds <= 0:
                messagebox.showwarning("Aviso", "O tempo deve ser maior que 0 segundos!", parent=popup)
                return
            
            popup.destroy()
            
            if modo_atual == 'aleatorio':
                mensagem = f"Resolvendo puzzle aleatório com:\n\n" + \
                          f"Algoritmo: {algoritmo}\n" + \
                          f"Tempo Máximo: {time_limit_seconds} segundos"
            else:
                mensagem = f"Resolvendo com:\n\n" + \
                          f"Algoritmo: {algoritmo}\n" + \
                          f"Tempo Máximo: {time_limit_seconds} segundos\n\n" + \
                          f"Imagem: {imagem_path}"
            
            messagebox.showinfo("Resolver Jogo", mensagem)
            print(matriz_atual)

            if isSolvable(matriz_atual):
                ordem="standard"
            else:
                ordem="backwards"
            
            if algoritmo == "A*":
                res_passos = resolucao_Astar(matriz_atual, time_limit_seconds,ordem)
            elif algoritmo == "BFS":
                res_passos = resolucao_BFS(matriz_atual, time_limit_seconds,ordem)
            elif algoritmo == "DFS":
                res_passos = resolucao_DFS(matriz_atual, time_limit_seconds,ordem)
            elif algoritmo == "GBFS":
                res_passos = resolucao_GBFS(matriz_atual, time_limit_seconds,ordem)
            elif algoritmo == "IDA*":
                res_passos = resolucao_IDAstar(matriz_atual, time_limit_seconds,ordem)
            elif algoritmo == "Reinforcement Learning":
                res_passos = resolucao_RL(matriz_atual,ordem)
            
            if res_passos is not None:
                mostrar_passos_popup(res_passos)
            else:
                messagebox.showwarning("Aviso", "O tempo não foi suficiente para resolver o problema")
            
        except ValueError:
            messagebox.showerror("Erro", "Não funcionou")
    
    def cancelar():
        """Fecha o popup sem fazer nada"""
        popup.destroy()
    
    Button(frame_btns, text="Resolver", command=executar_resolucao,
           bg="#4CAF50", fg="black", font="nunito 9 bold",
           padx=20, pady=5, cursor="hand2").grid(row=0, column=0, padx=5)
    
    Button(frame_btns, text="Cancelar", command=cancelar,
           bg="#f44336", fg="white", font="nunito 9",
           padx=20, pady=5, cursor="hand2").grid(row=0, column=1, padx=5)

def mostrar_passos_popup(res_passos):
    """Trata da exibição passo a passo da resolução"""
    if not res_passos:
        messagebox.showinfo("Info", "Não há passos para mostrar.")
        return

    popup = Toplevel(janela)
    popup.title("Resolução passo a passo")
    popup.geometry("600x500")
    popup.resizable(False, False)

    idx = [0]

    frame_matriz = Frame(popup)
    frame_matriz.pack(pady=10)

    lbl_info = Label(popup, text="", font=("Arial", 12))
    lbl_info.pack()

    def atualizar():
        for widget in frame_matriz.winfo_children():
            widget.destroy()

        atual = res_passos[idx[0]]
        proximo = res_passos[idx[0]+1] if idx[0] + 1 < len(res_passos) else None

        Label(frame_matriz, text="Estado Atual:", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=4, pady=5)
        for r, row in enumerate(atual):
            for c, val in enumerate(row):
                text = str(val) if val != 0 else ""
                bg = "#4CAF50" if val != 0 else "#e0e0e0"
                Label(frame_matriz, text=text, width=4, height=2, borderwidth=1, relief="solid", bg=bg).grid(row=r+1, column=c)

        if proximo:
            Label(frame_matriz, text="Seguinte:", font=("Arial", 10, "bold")).grid(row=6, column=0, columnspan=4, pady=5)
            for r, row in enumerate(proximo):
                for c, val in enumerate(row):
                    text = str(val) if val != 0 else ""
                    bg = "#81C784" if val != 0 else "#e0e0e0"
                    Label(frame_matriz, text=text, width=4, height=2, borderwidth=1, relief="solid", bg=bg).grid(row=r+7, column=c)

        lbl_info.config(text=f"Passo {idx[0]+1} de {len(res_passos)}")

    def ir_inicio():
        idx[0] = 0
        atualizar()

    def ir_fim():
        idx[0] = len(res_passos) - 1
        atualizar()

    def proximo_passos():
        if idx[0] < len(res_passos) - 1:
            idx[0] += 1
            atualizar()

    def anterior_passos():
        if idx[0] > 0:
            idx[0] -= 1
            atualizar()

    frame_botoes = Frame(popup)
    frame_botoes.pack(pady=20)
    Button(frame_botoes, text="⏮ Início", command=ir_inicio, width=10).grid(row=0, column=0, padx=5)
    Button(frame_botoes, text="◀ Anterior", command=anterior_passos, width=10).grid(row=0, column=1, padx=5)
    Button(frame_botoes, text="Próximo ▶", command=proximo_passos, width=10).grid(row=0, column=2, padx=5)
    Button(frame_botoes, text="Fim ⏭", command=ir_fim, width=10).grid(row=0, column=3, padx=5)

    atualizar()

def resolver_jogo():
    """Função para resolver o jogo"""
    if matriz_atual is not None:
        abrir_popup_algoritmo()
    else:
        messagebox.showwarning("Aviso", "Nenhuma matriz disponível! Selecione uma imagem ou gere uma matriz aleatória.")


# Main Window da aplicação
janela = Tk()
janela.title("Resolução Jogo dos 15")
janela.geometry("600x700") 
janela.resizable(False, False)

# --- Row 0 e 1: Título e Subtitulo ---
texto1 = Label(janela, text="Resolução do Jogo dos 15", font="nunito 16 bold")
texto1.grid(row=0, column=0, columnspan=4, pady=(20, 5))

texto2 = Label(janela, text="Insira uma imagem ou gere um puzzle aleatório:", font="nunito 10")
texto2.grid(row=1, column=0, columnspan=4, pady=(0, 10))

# --- Row 2: Botões de Seleção/Geração ---
frame_selecao = Frame(janela)
frame_selecao.grid(row=2, column=0, columnspan=4, pady=15)

btn_selecionar = Button(frame_selecao, text="Selecionar Imagem", 
                        command=selecionar_imagem,
                        font="nunito 10", bg="#4CAF50", fg="white",
                        padx=15, pady=8, cursor="hand2")
btn_selecionar.grid(row=0, column=0, padx=10)

btn_gerar_matriz = Button(frame_selecao, text="Gerar Puzzle Aleatório",
                        command=gerar_matriz_aleatoria,
                        font="nunito 10", bg="#9C27B0", fg="white",
                        padx=15, pady=8, cursor="hand2")
btn_gerar_matriz.grid(row=0, column=1, padx=10)

# --- Row 3: Visualização (Imagem ou Grid) ---
frame_imagem = Frame(janela, width=400, height=400, bg="gray90", relief=SUNKEN, bd=2)
frame_imagem.grid(row=3, column=0, columnspan=4, padx=100, pady=10)
frame_imagem.grid_propagate(False)

label_imagem = Label(frame_imagem, text="Nenhuma imagem ou puzzle selecionado", 
                     bg="gray90", font="nunito 9", fg="gray50")
label_imagem.place(relx=0.5, rely=0.5, anchor=CENTER)

# --- Row 4: Botões de Ação Final ---
frame_botoes = Frame(janela)
frame_botoes.grid(row=4, column=0, columnspan=4, pady=20)

btn_detetar = Button(frame_botoes, text="Detetar Disposição",
                     command=detetar_disposicao,
                     font="nunito 9", bg="#2196F3", fg="white",
                     padx=15, pady=10, cursor="hand2", state=DISABLED)
btn_detetar.grid(row=0, column=0, padx=10)

btn_resolver = Button(frame_botoes, text="Resolver o Jogo",
                      command=resolver_jogo,
                      font="nunito 9", bg="#FF9800", fg="white",
                      padx=15, pady=10, cursor="hand2", state=DISABLED)
btn_resolver.grid(row=0, column=1, padx=10)

janela.mainloop()
