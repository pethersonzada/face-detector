import cv2
import face_recognition
import os
import tkinter as tk
from tkinter import simpledialog
import numpy as np

# Função para garantir que uma pasta exista
def criar_pasta(pasta):
    if not os.path.exists(pasta):
        os.makedirs(pasta)

# Criar pastas necessárias
pasta_rostos_conhecidos = "Projeto/rostos_conhecidos"
criar_pasta(pasta_rostos_conhecidos)

# Função para salvar um rosto da câmera como conhecido
def salvar_novo_rosto():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar o vídeo. Tente novamente.")
            break

        cv2.imshow("Captura de Novo Rosto - Pressione 's' para salvar", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            root = tk.Tk()
            root.withdraw()
            nome_rosto = simpledialog.askstring("Input", "Digite o nome do novo rosto:")
            
            if nome_rosto:
                caminho_rosto = os.path.join(pasta_rostos_conhecidos, f"{nome_rosto}.jpg")
                imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(caminho_rosto, cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR))

                print(f"Rosto salvo como {nome_rosto}.jpg")

            break

    cap.release()
    cv2.destroyAllWindows()

# Inicialize o Tkinter para mostrar a janela de diálogo
root = tk.Tk()
root.withdraw()

# Perguntar ao usuário se deseja salvar um novo rosto
resposta = simpledialog.askstring("Input", "Deseja salvar um novo rosto? (sim/não)")

if resposta and resposta.lower() == "sim":
    salvar_novo_rosto()

# Inicialize a captura da webcam
cap = cv2.VideoCapture(0)

# Lista para codificações de rostos conhecidos e seus nomes
codificacoes_rostos_conhecidos = []
nomes_rostos_conhecidos = []

# Carregar rostos conhecidos de um diretório de imagens
for nome_arquivo in os.listdir(pasta_rostos_conhecidos):
    caminho_imagem = os.path.join(pasta_rostos_conhecidos, nome_arquivo)
    imagem = face_recognition.load_image_file(caminho_imagem)
    codificacoes_rostos_conhecidos.append(face_recognition.face_encodings(imagem)[0])
    nomes_rostos_conhecidos.append(os.path.splitext(nome_arquivo)[0])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o vídeo. Tente novamente.")
        break

    pequeno_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    frame_rgb_pequeno = cv2.cvtColor(pequeno_frame, cv2.COLOR_BGR2RGB)

    locais_rostos = face_recognition.face_locations(frame_rgb_pequeno)
    codificacoes_rostos = face_recognition.face_encodings(frame_rgb_pequeno, locais_rostos)

    for codificacao_rosto, local_rosto in zip(codificacoes_rostos, locais_rostos):
        correspondencias = face_recognition.compare_faces(codificacoes_rostos_conhecidos, codificacao_rosto)
        nome = "Desconhecido"

        distancias_rostos = face_recognition.face_distance(codificacoes_rostos_conhecidos, codificacao_rosto)
        indice_melhor_correspondencia = np.argmin(distancias_rostos)

        if correspondencias[indice_melhor_correspondencia]:
            nome = nomes_rostos_conhecidos[indice_melhor_correspondencia]

        topo, direita, baixo, esquerda = local_rosto
        topo *= 4
        direita *= 4
        baixo *= 4
        esquerda *= 4

        cv2.rectangle(frame, (esquerda, topo), (direita, baixo), (0, 255, 0), 2)
        cv2.putText(frame, nome, (esquerda, baixo + 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1)

    cv2.imshow('Reconhecimento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
