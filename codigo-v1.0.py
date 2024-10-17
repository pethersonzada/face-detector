# Importando bibliotecas.

import cv2
import os
import numpy as np


# Inicializa a webcam e retorna o objeto de captura.
def inicializar_webcam():
    return cv2.VideoCapture(0)


# Cria um diretório para as faces das pessoas.
def criar_diretorio_faces(nome_pessoa):

    diretorio_faces = 'face-detector/faces/'
    diretorio_pessoa = os.path.join(diretorio_faces, nome_pessoa)
    
    if not os.path.exists(diretorio_faces):
        os.makedirs(diretorio_faces)
    
    if not os.path.exists(diretorio_pessoa):
        os.makedirs(diretorio_pessoa)
    
    return diretorio_pessoa


# Captura imagens do rosto da pessoa e salva no diretório especificado.
def capturar_faces(webcam, diretorio_pessoa):

    classificador_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    contador = 0  # Contador para o número de imagens capturadas.
    
    while contador < 100:
        ret, quadro = webcam.read()  # Lê um quadro da webcam.
        if not ret:
            print("Falha ao capturar imagem.")
            break
            
        imagem_cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)  # Converte a imagem para escala de cinza.
        rostos = classificador_faces.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5)  # Detecta rostos.
        
        for (x, y, w, h) in rostos:
            imagem_rosto = imagem_cinza[y:y + h, x:x + w]  # Recorta a imagem do rosto.
            cv2.imwrite(os.path.join(diretorio_pessoa, f"{contador}.jpg"), imagem_rosto)  # Salva a imagem do rosto.
            contador += 1
            
        cv2.imshow("Captura de Rostos", quadro)  # Mostra o quadro com a captura.
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Treina o reconhecedor de rostos com as imagens armazenadas e retorna o reconhecedor e os IDs dos rótulos.
def treinar_reconhecedor(diretorio_faces):

    reconhecedor = cv2.face.LBPHFaceRecognizer_create()  # Cria o reconhecedor LBPH.
    rotulos = []  # Lista para armazenar rótulos (IDs).
    faces = []  # Lista para armazenar imagens de rostos.
    ids_rotulos = {}  # Dicionário para mapear IDs aos nomes.
    
    id_atual = 0
    for nome_pessoa in os.listdir(diretorio_faces):
        caminho_pessoa = os.path.join(diretorio_faces, nome_pessoa)
        
        for nome_imagem in os.listdir(caminho_pessoa):
            caminho_imagem = os.path.join(caminho_pessoa, nome_imagem)
            imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)  # Lê a imagem em escala de cinza.
            faces.append(imagem)  # Adiciona a imagem à lista de faces.
            rotulos.append(id_atual)  # Adiciona o ID atual à lista de rótulos.
        
        ids_rotulos[id_atual] = nome_pessoa  # Mapeia o ID ao nome da pessoa.
        id_atual += 1  # Incrementa o ID para a próxima pessoa.

    reconhecedor.train(faces, np.array(rotulos))  # Treina o modelo.
    reconhecedor.save("face-detector/face_trained.yml")  # Salva o modelo treinado.
    
    return reconhecedor, ids_rotulos


# Reconhece rostos usando o modelo treinado e exibe os resultados na tela.
def reconhecer_faces(webcam, reconhecedor, ids_rotulos):

    classificador_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, quadro = webcam.read()  # Lê um quadro da webcam.
        if not ret:
            print("Falha ao capturar imagem.")
            break
            
        imagem_cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)  # Converte a imagem para escala de cinza.
        rostos = classificador_faces.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5)  # Detecta rostos.

        for (x, y, w, h) in rostos:
            imagem_rosto = imagem_cinza[y:y + h, x:x + w]  # Recorta a imagem do rosto.
            rotulo, confianca = reconhecedor.predict(imagem_rosto)  # Tenta reconhecer o rosto.
            
            # Ajuste no limite de confiança.
            limite_confianca = 70  # Valor mais baixo significa maior confiança.
            if confianca < limite_confianca:  # Somente reconhece se a confiança for alta.
                nome = ids_rotulos.get(rotulo, "Desconhecido")  # Obtém o nome pelo ID.
                cor_texto = (255, 0, 0)  # Cor azul para rostos reconhecidos.
            else:
                nome = "Desconhecido"  # Se a confiança for baixa, considera como desconhecido.
                cor_texto = (0, 0, 255)  # Cor vermelha para rostos desconhecidos.
            
            # Adiciona texto ao quadro
            # Adicionar confiança para aparecer ao lado do nome:  - {confianca:.2f}
            
            cv2.putText(quadro, f'{nome}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_texto, 2)
            cv2.rectangle(quadro, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Desenha um retângulo ao redor do rosto.
        
        cv2.imshow("Reconhecimento Facial", quadro)  # Mostra o quadro com reconhecimento.
        
        if cv2.waitKey(5) == 27:  # Sai do loop se a tecla 'Esc' for pressionada.
            break


# Função principal que executa o fluxo do programa.
def main():
    while True:
        adicionar_pessoa = input("Deseja adicionar uma nova pessoa? (s/n): ").strip().lower()
        
        if adicionar_pessoa == 's':
            nome_pessoa = input("Digite o nome da pessoa: ")  # Solicita o nome da pessoa.
            webcam = inicializar_webcam()  # Inicializa a webcam.
            diretorio_pessoa = criar_diretorio_faces(nome_pessoa)  # Cria diretório para a pessoa.
            capturar_faces(webcam, diretorio_pessoa)  # Captura as faces da pessoa.
            webcam.release()  # Libera a webcam.
        elif adicionar_pessoa == 'n':
            break  # Sai do loop se o usuário não quiser adicionar mais pessoas.
        else:
            print("Opção inválida. Por favor, digite 's' ou 'n'.")

    reconhecedor, ids_rotulos = treinar_reconhecedor('face-detector/faces/')  # Treina o reconhecedor.
    
    # Carrega o modelo treinado, se ele existir.
    if os.path.exists("face-detector/face_trained.yml"):
        reconhecedor.read("face-detector/face_trained.yml")
    
    webcam = inicializar_webcam()  # Inicializa a webcam novamente.
    reconhecer_faces(webcam, reconhecedor, ids_rotulos)  # Inicia o reconhecimento de faces.
    webcam.release()  # Libera a webcam.
    cv2.destroyAllWindows()  # Fecha todas as janelas abertas.

if __name__ == "__main__":
    main()  # Executa a função principal.