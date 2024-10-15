import cv2 # Controlar webcam e carregar imagem da webcam.
import mediapipe as mp # Reconhecimento da imagem.

webcam = cv2.VideoCapture(0) # Usando webcam padrão (0) do computador/notebook.

solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

while True:
    verificador, frame = webcam.read()

    if not verificador:
        break

    lista_rostos = reconhecedor_rostos.process(frame)

    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(frame, rosto)

    cv2.imshow("Rostos", frame)

    if cv2.waitKey(5) == 27:
        break

webcam.release()