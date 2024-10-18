Este projeto implementa um sistema de reconhecimento facial utilizando duas abordagens distintas:

1. Reconhecimento com face_recognition: Utiliza a biblioteca face_recognition para detectar, codificar e reconhecer rostos em tempo real.


2. Reconhecimento com OpenCV, Mediapipe e LBPH: Utiliza o Mediapipe para detecção de rostos e o algoritmo LBPHFaceRecognizer do OpenCV para realizar o reconhecimento.



Funcionalidades

1. Reconhecimento Facial com face_recognition

Detecta e reconhece rostos a partir de imagens ou vídeo.

Permite adicionar novos rostos à base de dados em tempo real, capturando-os da webcam.

Salva as codificações dos rostos em uma pasta específica para futuras comparações.

Exibe o nome do rosto identificado ou "Desconhecido" se o rosto não for reconhecido.


2. Reconhecimento Facial com OpenCV, Mediapipe e LBPH

Detecta rostos em imagens capturadas pela webcam utilizando o Mediapipe.

Captura e armazena múltiplas imagens do rosto de cada pessoa para treinamento.

Treina um modelo de reconhecimento facial utilizando o algoritmo LBPH (Local Binary Patterns Histograms).

Permite o reconhecimento em tempo real, com uma taxa de confiança para determinar se o rosto é conhecido ou não.


Bibliotecas Usadas

face_recognition

face_recognition: Para codificação e comparação de rostos.

opencv-python: Para captura de vídeo e exibição dos resultados.

numpy: Para cálculos e manipulação de arrays.


OpenCV, Mediapipe e LBPH

opencv-python: Para captura de vídeo, processamento de imagens e reconhecimento facial LBPH.

mediapipe: Para detecção de rostos.

numpy: Para manipulação de dados numéricos.


Como Usar

1. Script com face_recognition

1. Instale as dependências:

pip install face_recognition opencv-python numpy


2. Execute o script para iniciar o reconhecimento facial:

python reconhecimento_face_recognition.py


3. Ao iniciar, você pode optar por adicionar um novo rosto, capturando a imagem pela webcam. Pressione 's' para salvar a imagem e insira o nome da pessoa.


4. O sistema irá reconhecer rostos em tempo real e exibir o nome das pessoas conhecidas ou "Desconhecido" para rostos não reconhecidos.



2. Script com OpenCV, Mediapipe e LBPH

1. Instale as dependências:

pip install opencv-python mediapipe numpy


2. Execute o script para adicionar novas pessoas e treinar o modelo:

python reconhecimento_opencv_lbph.py


3. O sistema irá solicitar se você deseja adicionar uma nova pessoa. Caso sim, o script capturará 100 imagens do rosto para treinar o modelo LBPH.


4. Após o treinamento, o sistema estará pronto para reconhecer rostos em tempo real, exibindo o nome da pessoa ou "Desconhecido".



Estrutura do Projeto

.
├── reconhecimento_face_recognition.py  # Script com face_recognition
├── reconhecimento_opencv_lbph.py       # Script com OpenCV, Mediapipe e LBPH
├── face-detector/                      # Diretório contendo os dados de treinamento LBPH
│   ├── faces/                          # Imagens capturadas para treinamento
│   └── face_trained.yml                # Modelo treinado LBPH
└── Projeto/
    └── rostos_conhecidos/              # Imagens de rostos salvos pelo face_recognition

Contribuição

1. Faça um fork deste repositório.


2. Crie uma nova branch para suas alterações.


3. Envie um pull request com uma descrição detalhada das suas alterações.



Licença

Este projeto é licenciado sob os termos da licença MIT.

