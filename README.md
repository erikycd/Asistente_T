# Entrenamiento del Asistente de aprendizaje

En este repositorio se muestra el proceso para generar los modelos de IA capaces de entrenar al [Asistente de aprendizaje](https://github.com/erikycd/Asistente_A) en algunas de las siguientes habilidades:

* Lógica del agente (también llamado Dialog manager): Encargada del flujo de información mediante un modelo de clasificación,
* Conversacional: Encargada de la habilidad conversacional mediante el modelo de lenguaje GPT2 de Huggingface.

Tabla de contenido:

0. [Instalación](https://github.com/erikycd/Asistente_T#instalaci%C3%B3n-creaci%C3%B3n-del-ambiente-e-instalaci%C3%B3n-de-dependencias)

1. [Entrenamiento de la Lógica del agente](https://github.com/erikycd/Asistente_T#1-sobre-la-l%C3%B3gica-del-agente)

3. [Entrenamiento conversacional](https://github.com/erikycd/Asistente_T#2-sobre-las-funciones-conversacionales)

## Instalación. Creación del ambiente e instalación de dependencias

Descargar manualmente o clonar este repositorio mediante:
```
$ git clone https://github.com/erikycd/Asistente_T.git
```
Crear un ambiente de [Anaconda](https://www.anaconda.com/distribution/) con todas las librerias por defecto de python v.3.9 a través del siguiente comando en Anaconda Prompt:
```
$ conda create --name asistente python==3.9
```
Activar el ambiente recien creado:
```
$ conda activate asistente
```
Cambiar el directorio de trabajo a la carpeta donde el asistente ha sido alojado:
```
$ cd C:\Users\erikcd\Documents\Asistente_T
```
Instalar las librerias particulares que requiere el asistente a través del archivo `requirements.txt` con el siguiente comando:
```
$ pip install -r requirements.txt
```

## 1. Entrenamiento de la lógica del agente

### 1.1 Proceso de entrenamiento
El proceso de entrenamiento genera un modelo de clasificación mediante el algoritmo de Random Forest, para los datos en el archivo `intent_file_word2vec.yml`, el cual contiene las etiquetas (clases) e intenciones por cada clase. Ver siguiente figura:

<p align="center">
  <img width="45.0%" src="https://github.com/erikycd/Asistente_T/blob/main/image_intent.png?raw=true">
</p>

Ejecutar el programa principal con el siguiente comando:
```
$ python main_word2vec_training.py
```
Finalmente, el modelo entrenado `model_word2vec.sav` y un reporte de entrenamiento `report.txt` son guardados en la carpeta `./word2vec_engine`. Un ejemplo del espacio de clasificación de 3 dimensiones se puede ver en la siguiente figura:

<p align="center">
  <img width="45.0%" src="https://github.com/erikycd/Asistente_T/blob/main/word2vec_engine/word2vec_figure.png?raw=true">
</p>

### 1.2 Proceso de inferencia

## 2. Entrenamiento conversacional

### 2.1 Proceso de entrenamiento

### 2.2 Proceso de inferencia
