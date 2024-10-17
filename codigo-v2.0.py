import cv2
import os
import numpy as np
import mediapipe as mp

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
    contador = 0
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    while contador < 100:
        ret, quadro = webcam.read()
        if not ret:
            print("Falha ao capturar imagem.")
            break
        
        imagem_rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
        resultados = face_detection.process(imagem_rgb)
        
        if resultados.detections:
            for detection in resultados.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = quadro.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Verifica se a caixa delimitadora está dentro dos limites da imagem
                if bbox[1] >= 0 and bbox[0] >= 0 and (bbox[1] + bbox[3]) <= h and (bbox[0] + bbox[2]) <= w:
                    imagem_rosto = quadro[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                    cv2.imwrite(os.path.join(diretorio_pessoa, f"{contador}.jpg"), imagem_rosto)
                    contador += 1

        cv2.imshow("Captura de Rostos", quadro)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Treina o reconhecedor de rostos com as imagens armazenadas e retorna o reconhecedor e os IDs dos rótulos.
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

# Reconhece rostos usando o modelo treinado e exibe os resultados na tela.
def reconhecer_faces(webcam, reconhecedor, ids_rotulos):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    while True:
        ret, quadro = webcam.read()
        if not ret:
            print("Falha ao capturar imagem.")
            break
        
        imagem_rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
        resultados = face_detection.process(imagem_rgb)

        if resultados.detections:
            for detection in resultados.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = quadro.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Verifica se a caixa delimitadora está dentro dos limites da imagem
                if bbox[1] >= 0 and bbox[0] >= 0 and (bbox[1] + bbox[3]) <= h and (bbox[0] + bbox[2]) <= w:
                    imagem_rosto = cv2.cvtColor(quadro[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]], cv2.COLOR_BGR2GRAY)
                    rotulo, confianca = reconhecedor.predict(imagem_rosto)
                    
                    limite_confianca = 60
                    
                    if confianca < limite_confianca:
                        nome = ids_rotulos.get(rotulo, "Desconhecido")
                        cor_texto = (255, 0, 0)  # Azul
                    else:
                        nome = "Desconhecido"
                        cor_texto = (0, 0, 255)  # Vermelho
                    
                    cv2.putText(quadro, f'{nome}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_texto, 2)
                    cv2.rectangle(quadro, bbox, (255, 0, 0), 2)  # Desenha um retângulo ao redor do rosto.
        
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
