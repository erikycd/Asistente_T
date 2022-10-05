# Entrenamiento del Asistente de aprendizaje

En este repositorio se muestra el proceso para generar los modelos de IA capaces de entrenar al [Asistente de aprendizaje](https://github.com/erikycd/Asistente_A) en algunas de las siguientes habilidades:

* Lógica del agente (también llamado Dialog manager): Encargada del flujo de información mediante un modelo de clasificación
* Conversacional: Encargada de la habilidad conversacional mediante el modelo de lenguaje GPT2 de Huggingface.

Tabla de contenido:

0. [Instalación]()

1. [Lógica del agente]()

3. [Convesacional]()

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

## 1. Sobre la lógica del agente

### * Proceso de entrenamiento

### * Proceso de inferencia

## 2. Sobre las funciones conversacionales

### * Proceso de entrenamiento

### * Proceso de inferencia
