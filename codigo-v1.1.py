import cv2
import os
import numpy as np

def inicializar_webcam():
    return cv2.VideoCapture(0)

def criar_diretorio_faces(nome_pessoa):
    diretorio_faces = 'face-detector/faces/'
    diretorio_pessoa = os.path.join(diretorio_faces, nome_pessoa)
    
    if not os.path.exists(diretorio_faces):
        os.makedirs(diretorio_faces)
    
    if not os.path.exists(diretorio_pessoa):
        os.makedirs(diretorio_pessoa)
    
    return diretorio_pessoa

def capturar_faces(webcam, diretorio_pessoa):
    classificador_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    contador = 0
    
    while contador < 100:
        ret, quadro = webcam.read()
        if not ret:
            print("Falha ao capturar imagem.")
            break
            
        imagem_cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
        rostos = classificador_faces.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in rostos:
            imagem_rosto = imagem_cinza[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(diretorio_pessoa, f"{contador}.jpg"), imagem_rosto)
            contador += 1
            
        cv2.imshow("Captura de Rostos", quadro)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def treinar_reconhecedor(diretorio_faces):
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    rotulos = []
    faces = []
    ids_rotulos = {}
    
    id_atual = 0
    for nome_pessoa in os.listdir(diretorio_faces):
        caminho_pessoa = os.path.join(diretorio_faces, nome_pessoa)
        
        for nome_imagem in os.listdir(caminho_pessoa):
            caminho_imagem = os.path.join(caminho_pessoa, nome_imagem)
            imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
            faces.append(imagem)
            rotulos.append(id_atual)
        
        ids_rotulos[id_atual] = nome_pessoa
        id_atual += 1

    reconhecedor.train(faces, np.array(rotulos))
    reconhecedor.save("face-detector/face_trained.yml")
    
    return reconhecedor, ids_rotulos

def reconhecer_faces(webcam, reconhecedor, ids_rotulos):
    classificador_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, quadro = webcam.read()
        if not ret:
            print("Falha ao capturar imagem.")
            break
            
        imagem_cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
        rostos = classificador_faces.detectMultiScale(imagem_cinza, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in rostos:
            imagem_rosto = imagem_cinza[y:y + h, x:x + w]
            rotulo, confianca = reconhecedor.predict(imagem_rosto)
            
            limite_confianca = 60
            
            # Verifica se a confiança é maior que o limite
            if confianca < limite_confianca:
                nome = ids_rotulos.get(rotulo, "Desconhecido")
                # Aqui você pode adicionar um código para calcular a distância de reconhecimento
                # se necessário, como comparar características faciais ou aplicar um threshold adicional
                cor_texto = (255, 0, 0)  # Azul
            else:
                nome = "Desconhecido"
                cor_texto = (0, 0, 255)  # Vermelho
            
            # Exibe o nome e desenha o retângulo ao redor do rosto
            cv2.putText(quadro, f'{nome}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_texto, 2)
            cv2.rectangle(quadro, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imshow("Reconhecimento Facial", quadro)
        
        if cv2.waitKey(5) == 27:  # Sai do loop se a tecla 'Esc' for pressionada.
            break

def main():
    while True:
        adicionar_pessoa = input("Deseja adicionar uma nova pessoa? (s/n): ").strip().lower()
        
        if adicionar_pessoa == 's':
            nome_pessoa = input("Digite o nome da pessoa: ")
            webcam = inicializar_webcam()
            diretorio_pessoa = criar_diretorio_faces(nome_pessoa)
            capturar_faces(webcam, diretorio_pessoa)
            webcam.release()
        elif adicionar_pessoa == 'n':
            break
        else:
            print("Opção inválida. Por favor, digite 's' ou 'n'.")

    reconhecedor, ids_rotulos = treinar_reconhecedor('face-detector/faces/')
    
    if os.path.exists("face-detector/face_trained.yml"):
        reconhecedor.read("face-detector/face_trained.yml")
    
    webcam = inicializar_webcam()
    reconhecer_faces(webcam, reconhecedor, ids_rotulos)
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
