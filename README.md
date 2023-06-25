# Preparacion del ambiente y librerias

## Instalacion de Open 3D

Open 3D es una biblioteca de código abierto que admite el desarrollo rápido de software que trata con datos 3D.
http://www.open3d.org/ 

```ruby

!pip install open3d;

```

## Librerias Usadas

```ruby

import csv
import open3d as o3d
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

```

## Inicializacion de variables 

Aqui se inician las variables principales del codigo, incluyendo el numero de epocas, el numero de datos que se tienen en el dataset, la cantidad de labels y la direccion donde se encuentran los datos en drive. 

# Crear Listas

## Crear nube de puntos de una malla 

Cada nube de puntos corresponde a la recosntruccion de una planta, cuando se etiquetan los datos, cada etiqueta se separa en un archivo con el formato: NombreArchivo_000001 dando como resultado un conjunto como el siguiente.

![Creacion de Mascara](https://github.com/Brayanjurado1325/Segmentacion-de-datos/blob/main/Imagenes/1.png)

Para volver a unir una nube de puntos se ingresa a la carpeta que la contiene (cada carpeta tiene el nombre de la nube que es un valor numerico para que se puedan recorrer) el algoritmo cuenta cuantos archivos .txt tiene la carpeta y concatena los datos.


## Visualizacion de nube de puntos
Al etiquetar los datos, la etiqueta de cada punto depende del organo de la planta al que pertenecen, se etiqueda con numeros del 0 a 4 de la siguiente manera.

0 No etiquetado
1 hojas
2 tierra
3 panícula
4 tallo
La entrada de la funcion son dos arrays:

point_cloud que corresponde a los las coordenadas XYZ de cada punto y labels que contiene la etiqueda de 0 a 4.

La funcion lee cada uno de los puntos, los imprime y da color dependiendo de la etiqueta.

Esta funcion se tomo de https://github.com/gitDux/Panicle-3D


![Creacion de Mascara](https://github.com/Brayanjurado1325/Segmentacion-de-datos/blob/main/Imagenes/2.png)


## Creacion de One-HOT

Para el algoritmo descrito en https://keras.io/examples/vision/pointnet_segmentation/
una de las entradas para el modelo es la etiqueta de los datos en formato one- hot donde el valor de la etiqueta que esta entre 0 y 4 se convierte en un vector que de tamaño 5 compuesto por ceros y un unico valor alto de 1 ubicao en la posicion segun corresponda el valor entero de la etiqueta asi:

* 0 es [1 0 0 0 0]
* 1 es [0 1 0 0 0]
* 2 es [0 0 1 0 0]
* 3 es [0 0 0 1 0]
* 4 es [0 0 0 0 1]


## Depuracion de datos

La etiqueta 0 hace referencia a los datos que durante el proceso de etiquetado no hacen parte de ningun organo de la planta, como se observa en la imagen de color azul, esto hace parte del ruido de la planta, entonces la primer parte del codigo elimina estos puntos de la nube de puntos de la planta, teniendo un resultado como el siguiente.


![Creacion de Mascara](https://github.com/Brayanjurado1325/Segmentacion-de-datos/blob/main/Imagenes/3.png)


## Crear Listas

Para crear la data de entrenamiento y validacion se deben organizar todas las nubes de puntos y etiquetas en Listas, se crean 3 listas, cada componente de la lista es un array:

**point_clouds**: Contiene las coordenadas de las nubes de puntos XYZ.

**all_labels**: Contiene la etiquetas de las nubes de puntos en formato enteo de 0 a 4.

**point_cloud_labels**: Contiene ña etiqueta de las nubes de punto en formato One-Hot

Inicialmente se llama la funcion de cargarnube para llamar cada nube de puntos, que trae los siguientes datos:

![Creacion de Mascara](https://github.com/Brayanjurado1325/Segmentacion-de-datos/blob/main/Imagenes/4.png)

Posteriormente se toman los primeros 3 datos para obtener las coordenadas y el ultimo dato para obtener la etiqueta de la nube de puntos, posteriormente se hace la depuracion de la nube de puntos para agregar la nube a la listas point_clouds y all_labels, finalmente se obtiene el array one-hot para agregarlo a la lista point_cloud_labels.


# Preprocesamiento y TensorFlow

## Muestreo y Normalizacion 

```ruby

for index in tqdm(range(len(point_clouds))):
    current_point_cloud = point_clouds[index]
    current_label_cloud = point_cloud_labels[index]
    current_labels = all_labels[index]
    num_points = len(current_point_cloud)
    # Muestreo aleatorio de los respectivos índices..
    sampled_indices = random.sample(list(range(num_points)), NUM_SAMPLE_POINTS)
    # Puntos de muestreo correspondientes a los índices muestreados.
    sampled_point_cloud = np.array([current_point_cloud[i] for i in sampled_indices])
    # Muestreo de etiquetas codificadas one - hot correspondientes.
    sampled_label_cloud = np.array([current_label_cloud[i] for i in sampled_indices])
    # Muestreo de etiquetas correspondientes para visualización.
    sampled_labels = np.array([current_labels[i] for i in sampled_indices])
    # Normalización de la nube de puntos muestreada.
    norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud, axis=0)
    norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
    point_clouds[index] = sampled_point_cloud
    #point_clouds[index] = norm_point_cloud
    point_cloud_labels[index] = sampled_label_cloud
    all_labels[index] = sampled_labels

```

![Creacion de Mascara](https://github.com/Brayanjurado1325/Segmentacion-de-datos/blob/main/Imagenes/5.png)



# Creacion de Modelo 


## Implementacion de PointNet 


## Instancias del modelo 


## Visualizacion de resultados. 


![Creacion de Mascara](https://github.com/Brayanjurado1325/Segmentacion-de-datos/blob/main/Imagenes/6.png)


![Creacion de Mascara](https://github.com/Brayanjurado1325/Segmentacion-de-datos/blob/main/Imagenes/7.png)














