# Projeto AIPI

Projeto desenvolvido nas Unidades Curriculares de **Processamento de Imagem (PI)** e **Inteligência Artificial para Sistemas Autónomos (IASA)**.

---

## Autores

- Diogo Ruas
- Afonso Pereira
- José Ribeiro

---

## Classificação

- **IASA:** 16 valores  (0-20)
- **PI:** 17.1 valores  (0-20)

---

## Run & Go

### Requirements

Instalar as dependências necessárias:

```bash
pip install opencv-python
pip install Pillow
pip install tk
pip install matplotlib
```
### Caso o computador suporte CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Utilizar PyTorch com CPU

```bash
pip install torch torchvision
```

---

## Executar a aplicação

```bash
git clone https://github.com/josemribeiro/projetoAIPI.git
cd projetoAIPI
python AppMainLoop.py
```

---

## Notas sobre o Projeto

As imagens utilizadas para testes encontram-se na pasta **`15GameImages`**, onde o método de **template matching** funciona de forma consistente.

O projeto também utiliza **Redes Neuronais Convolucionais (CNN)** para auxiliar no processamento e análise das imagens, no entanto esta abordagem apresenta algumas limitações.

Algumas limitações identificadas:

- Os resultados obtidos através da **CNN podem nem sempre ser os esperados**, dependendo das condições das imagens.

- A **máscara utilizada para obter a imagem binária do tabuleiro** não é totalmente robusta, podendo apresentar dificuldades em **imagens mais escuras ou com iluminação irregular**.
