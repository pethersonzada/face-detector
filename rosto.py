import cv2
import os
import numpy as np

# Inicializa a webcam
webcam = cv2.VideoCapture(0)

# Inicializa o classificador de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Diretório para salvar as imagens capturadas
face_dir = 'faces/'
if not os.path.exists(face_dir):
    os.makedirs(face_dir)

# Nome da pessoa sendo capturada
person_name = input("Digite o nome da pessoa: ")
person_dir = face_dir + person_name
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

count = 0
while True:
    ret, frame = webcam.read()
    
    # Converte para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta rostos
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Desenha um retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Salva a imagem do rosto
        face_img = gray[y:y + h, x:x + w]
        cv2.imwrite(f"{person_dir}/{count}.jpg", face_img)
        count += 1

    cv2.imshow("Captura de Rostos", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50 : # Limita para 50 imagens.
        break

webcam.release()
cv2.destroyAllWindows()

# Caminho para as imagens
face_dir = 'faces/'

# Inicializa o LBPH reconhecedor
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Inicializa o classificador de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Lista para armazenar dados de treino
labels = []
faces = []
label_ids = {}
current_id = 0

# Carrega todas as imagens e associa com IDs
for person_name in os.listdir(face_dir):
    person_path = os.path.join(face_dir, person_name)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        face = img
        faces.append(face)
        labels.append(current_id)
    
    label_ids[current_id] = person_name
    current_id += 1

# Treina o modelo com as imagens e IDs
recognizer.train(faces, np.array(labels))

# Salva o modelo treinado
recognizer.save("face_trained.yml")

# Carrega o modelo treinado
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trained.yml")

# Inicializa o classificador de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mapeamento de IDs para nomes
label_ids = {}  # Inicializa o dicionário para armazenar o mapeamento
current_id = 0

# Diretório das faces
face_dir = 'faces/'

# Carrega todas as imagens e associa com IDs
for person_name in os.listdir(face_dir):
    person_path = os.path.join(face_dir, person_name)
    if os.path.isdir(person_path):  # Verifica se é um diretório
        label_ids[current_id] = person_name  # Mapeia o ID para o nome da pessoa
        current_id += 1

# Inicia a webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        
        # Tenta reconhecer o rosto
        label, confidence = recognizer.predict(face_img)
        
        # Mostra o nome e confiança do rosto reconhecido
        if confidence < 100:
            name = label_ids.get(label, "Desconhecido")
            cv2.putText(frame, f'{name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, 'Desconhecido', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Desenha um retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow("Reconhecimento Facial", frame)
    
    if cv2.waitKey(5) == 27:  # Pressione 'Esc' para sair
        break

webcam.release()
cv2.destroyAllWindows()
