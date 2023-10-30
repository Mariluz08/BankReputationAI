##install pandas scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
 
# Cargar datos
##data = pd.read_csv('tu_archivo.csv')
data =  pd.read_csv(open('pruebabco2.csv'))
# Visualizar los primeros registros
########print(data.head())
##Paso 4: Preprocesamiento de Datos
##python
##Copy code
#####df['Hora'] = pd.to_datetime(df['Hora'])
#df['Hora'] = pd.to_datetime(df['Hora'])
#df['Hora'] = df['Hora'].dt.round('480min')
count_df = df.groupby(['Hora','Sentimiento']).size().unstack(fill_value=0)
##print(df)
print(count_df)
##print(data.head())
# Dividir datos en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.5, random_state=42)
 
# Crear vectores TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train = tfidf_vectorizer.fit_transform(train_data['Contenido'])
X_test = tfidf_vectorizer.transform(test_data['Contenido'])
 
# Etiquetas
y_train = train_data['Sentimiento']
y_test = test_data['Sentimiento']
###Paso 5: Entrenar el Modelo
##python
##Copy code
# Inicializar el clasificador 
classifier = MultinomialNB()
 
# Entrenar el modelo
classifier.fit(X_train, y_train)
###Paso 6: Evaluación del Modelo
##python
##Copy code
# Predecir en el conjunto de prueba
y_pred = classifier.predict(X_test)
 
# Evaluar el rendimiento
print("Exactitud:", accuracy_score(y_test, y_pred))
print("Informe de clasificación:\n", classification_report(y_test, y_pred))
 
##Paso 7: Uso del Modelo
##python
##Copy code
# Puedes usar el modelo para predecir el sentimiento de nuevos textos
##new_texts = ["recomendado", "muy bien.","buen servicio"]
new_texts_vectorized = tfidf_vectorizer.transform(new_texts)
predictions = classifier.predict(new_texts_vectorized)
print("Predicciones:", predictions)
