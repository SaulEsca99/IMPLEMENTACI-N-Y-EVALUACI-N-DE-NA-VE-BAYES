# 📊 PRÁCTICA 3: IMPLEMENTACIÓN Y EVALUACIÓN DE NAÏVE BAYES
## Tecnologías de Lenguaje Natural

**Autor:** Escamilla Lazcano Saúl
**Carrera:** Ingeniería En Inteligencia Artificial

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Librería-Pandas-purple.svg)](https://pandas.pydata.org/)
[![spaCy](https://img.shields.io/badge/Librería-spaCy-blue.svg)](https://spacy.io/)
[![NLTK](https://img.shields.io/badge/Librería-NLTK-green.svg)](https://www.nltk.org/)
[![Scikit-learn](https://img.shields.io/badge/Librería-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Librería-Seaborn%20%7C%20Matplotlib-blueviolet.svg)](https://seaborn.pydata.org/)

## 🚀 Descripción del Proyecto

Este proyecto es una implementación completa del algoritmo clasificador **Bayesiano Ingenuo (Naïve Bayes)** **desde cero** en Python. El objetivo es construir un modelo de **Análisis de Sentimiento** capaz de clasificar reseñas de películas como "positivas" o "negativas" basándose únicamente en su contenido textual.

El *pipeline* del proyecto cubre todos los pasos esenciales de una tarea de PLN:
1.  **Carga y Exploración de Datos** del dataset IMDB.
2.  **Preprocesamiento y Normalización de Texto** avanzado usando `spaCy` y `NLTK`.
3.  **Implementación del Modelo** (`NaiveBayesPersonalizado`) desde cero.
4.  **Entrenamiento y Evaluación** del modelo con métricas de clasificación estándar.
5.  **Visualización de Resultados**, incluyendo una matriz de confusión y nubes de palabras.

## 💾 1. Dataset

Se utiliza el **"IMDB Dataset of 50K Movie Reviews"**. Este es un corpus canónico para tareas de clasificación binaria de sentimiento.
* **Archivo:** `IMDB Dataset.csv`
* **Tamaño:** 50,000 reseñas.
* **Clases:** "positiva" (25,000) y "negativa" (25,000).

## ⚙️ 2. Pipeline de Preprocesamiento de Texto

Para que el modelo pueda "entender" el texto, este debe ser normalizado. Se implementó un pipeline robusto que seleccionó la **Lematización (con `spaCy`)** sobre el *Stemming* (con `NLTK`), ya que la lematización produce palabras léxicamente correctas (lemas), lo cual es más preciso.

El pipeline de normalización (`lematizar_texto`) realiza los siguientes pasos:
1.  **Conversión a Minúsculas:** `texto.lower()`
2.  **Eliminación de HTML:** Se utiliza `re` para eliminar etiquetas (ej. `<br />`).
3.  **Tokenización (spaCy):** Se procesa el texto con el modelo `en_core_web_sm`.
4.  **Filtrado de Tokens:** Se eliminan *stopwords*, puntuación y espacios.
5.  **Lematización (spaCy):** Cada token se reduce a su forma base de diccionario (ej. "running" → "run").

## 🧠 3. Implementación del Modelo: `NaiveBayesPersonalizado` (Desde Cero)

El núcleo de la práctica es esta clase, que no utiliza las implementaciones de `sklearn` para el clasificador.

### A. Entrenamiento (`fit`)
El método `fit` aprende las probabilidades necesarias del corpus de entrenamiento (`X_train`, `y_train`).

**1. Cálculo de Priors de Clase $P(c)$:**
Calcula la probabilidad base de cada clase (positiva o negativa) en el dataset.
$$ P(c) = \frac{\text{Documentos en la clase } c}{\text{Total de documentos}} $$

**2. Cálculo de Probabilidades Condicionales (Likelihoods) $P(w|c)$:**
Calcula la probabilidad de que una palabra $w$ aparezca, dado que pertenece a una clase $c$.

*
