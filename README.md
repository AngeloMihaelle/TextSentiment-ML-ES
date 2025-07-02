# 📊 Proyecto: Evaluación de Modelos de Clasificación de Texto para Opiniones en Español 📝

## 📖 Descripción

Este proyecto consiste en comparar y evaluar diferentes modelos de clasificación automática de texto para predecir la **polaridad** de opiniones en español. Utiliza un dataset de reseñas mexicanas (`Rest_Mex_2022.xlsx`) que contiene columnas de título, opinión y su respectiva polaridad.

Se comparan diferentes representaciones de texto (Binarizada, Frecuencia y TF-IDF) junto con varios algoritmos de clasificación supervisada:

* 🤖 Regresión Logística
* ⚫ Máquina de Vectores de Soporte (SVM)
* 🌲 Bosques Aleatorios (Random Forest)
* 📊 Naive Bayes Multinomial

El objetivo es encontrar la mejor combinación de representación y modelo basada en la métrica **F1 macro** utilizando validación cruzada.

---

## 🗂 Estructura del Proyecto

* `practica_6_evalModels.py`: Script principal que realiza la carga, preprocesamiento, entrenamiento y evaluación de los modelos.

* `Rest_Mex_2022.xlsx`: Dataset en formato Excel con las opiniones y etiquetas de polaridad. (Debe estar en el mismo directorio o especificar ruta correcta)

---

## ⚙️ Detalles Técnicos

### 📥 Carga y Preprocesamiento de Datos

* Se usa `pandas` para cargar el archivo Excel.
* Se concatenan las columnas `Title` y `Opinion` para formar el texto completo a analizar.
* No se aplica preprocesamiento adicional más allá de la concatenación de textos (en este código específico).
* El texto sin procesar se usa para la vectorización posterior.

### 📝 Representaciones Textuales

Se prueban tres formas de representar el texto en números:

* ✔️ **Binarizada:** Matriz de presencia/ausencia de palabras (`CountVectorizer` con `binary=True`).
* 🔢 **Frecuencia:** Conteo normal de palabras (`CountVectorizer` estándar).
* 📈 **TF-IDF:** Peso que combina frecuencia inversa de documento (`TfidfVectorizer`).

### 🤖 Modelos Probados

* 🧮 **Regresión Logística:** Modelo lineal para clasificación.
* ⚫ **SVM:** Máquinas de vectores de soporte.
* 🌲 **Random Forest:** Ensamble de árboles de decisión.
* 📊 **Naive Bayes Multinomial:** Basado en probabilidades de palabras.

### 🧪 Evaluación

* Se realiza una división estratificada 80/20 para entrenamiento y prueba.
* Se emplea validación cruzada de 5 folds sobre el conjunto de entrenamiento.
* La métrica utilizada para comparar modelos es **F1 macro**, que considera el balance entre precisión y recall para todas las clases.
* Se selecciona el modelo con mejor rendimiento promedio en la validación cruzada.
* Finalmente, se evalúa el modelo seleccionado en el conjunto de prueba y se muestran:

  * 📋 Reporte de clasificación detallado (precisión, recall, f1-score por clase)
  * 🔍 Matriz de confusión

---

## 🛠 Requisitos

* Python 3.7 o superior
* Librerías Python:

  * pandas
  * scikit-learn
  * spacy
  * openpyxl (para lectura de Excel)
* Modelo de spaCy para español:

  ```bash
  python -m spacy download es_core_news_sm
  ```

---

## 🚀 Instrucciones para ejecutar

1. Clona o descarga este repositorio.
2. Asegúrate de tener instalado Python y las librerías requeridas:

   ```bash
   pip install pandas scikit-learn spacy openpyxl
   python -m spacy download es_core_news_sm
   ```
3. Coloca el archivo `Rest_Mex_2022.xlsx` en la carpeta del proyecto.
4. Ejecuta el script:

   ```bash
   python practica_6_evalModels.py
   ```
5. Observa en consola la evaluación de todas las combinaciones y el resultado final con el mejor modelo.

---

## 📊 Resultados esperados

* Impresión en consola de las combinaciones vectorizador + modelo con su respectiva media de F1 macro en validación cruzada.
* Detalle del modelo seleccionado.
* Evaluación final con métricas y matriz de confusión para el conjunto de prueba.

---

## 🔧 Posibles mejoras

* ✂️ Aplicar preprocesamiento de texto (lemmatización, eliminación de stopwords).
* 🎯 Afinar hiperparámetros mediante búsqueda en malla (GridSearchCV).
* 📈 Añadir otras métricas de evaluación.
* 💾 Guardar el mejor modelo para uso posterior.
* 📊 Visualización gráfica de resultados.

---

## 👤 Autor

Proyecto desarrollado por Angelo Ojeda.


