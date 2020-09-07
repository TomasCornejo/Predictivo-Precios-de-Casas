# Precios de Casas - Predictivo <img src="https://github.com/TomasCornejo/PredictivoVariableContinua/blob/master/img/House%20Logo.jpg" width="60" height="60" />

Este trabajo es un modelo predictivo para una variable continua usando una base extraida de las competiciones Kaggle.

El dataset está conformado por información de casas donde las variables independientes son areas, cantidades y diferentes clasificaciones, mientras que la variable dependiente es el precio de la misma.

La principal idea es plantear el problema y mostrar como es la forma de trabajo que utilizo para solucionarlo.

## Metodología

Para desarrollar esta solución se seguiran las siguiente metodología:

![Flujo](https://github.com/TomasCornejo/PredictivoVariableContinua/blob/master/img/FlujoDeTrabajo.jpg)

### Análisis Descriptivo

En este paso se generaron dos reportes usando pandas-profiling:

Para el Dataset de [Entrenamiento](https://github.com/TomasCornejo/Predictivo-Precios-de-Casas/blob/master/train_ds.html) y para el de [Testeo](https://github.com/TomasCornejo/Predictivo-Precios-de-Casas/blob/master/test_ds.html) (Para visualización de los reportes descargar)

Lo anterior permitió generar descripciones de variables como la siguiente:
![](https://github.com/TomasCornejo/Predictivo-Precios-de-Casas/blob/master/img/Descripcion_de_Variables.PNG)

y la matriz de correlación para variables numéricas en un comienzo:
![](https://github.com/TomasCornejo/Predictivo-Precios-de-Casas/blob/master/img/MatrizCorrelatcion.PNG)

### Preprocesamiento de Datos

Para preprocesar el dataset se separó en 3 tipos de datos:

-Variables Numéricas

-Variables Categóricas Ordinales(que seran convertidas mediante diccionarios dependiendo de cada caso)

-Variables Categóricas No Ordinales(en los que se usara OneHotEncoding para convertir a variables binarias)

Luego de lo anterior se generó un dataset de 211 variables con las que se generarán los modelos.

### Baseline

Modelo creado a partir solo de las variables numéricas que tienen alta correlación con la variable dependiente, a modo Baseline para tener un punto de partida.
Los resultadosen la primera entrega son:

```
Linear        | MSLE = 0.033

Tree          | MSLE = 0.047

RandomForest  | MSLE = 0.025

MultiLayer    | MSLE = 0.045
```

### Tuning

Luego del baseline se generaro una serie de modelos para ajustar los hiperparametros del RandomForest dado que obtuvo el mejor rendimiento en el baseline:

```
n_estimators = list(range(10,110,10))
max_depth = list(range(10,110,20))
min_samples_leaf = list(range(1,20,4))
max_features = list(range(20,211,15))

#Iteracion de parametros para encontrar el optimo Random Forest
for n_est in n_estimators:
    for md in max_depth:
        for msl in min_samples_leaf:
            for mf in max_features:
                name = f'RandomForest_{n_est}_{md}_{msl}_{mf}'
                model = RFR(n_estimators=n_est , max_depth = md , min_samples_leaf = msl, max_features= mf)
                Modelos.append([name,model])
                print(f'Modelo: {name}\\')

```
Lo anterior sumado a la inclusión de las variables categóricas, permitió obtener:

```
RandomForest_40_90_5_80 - MSLE = 0.024

```
La mejora fue leve respecto a la etapa anterior, por lo que se exploró otra técnica de Ensemble, mas especificamente el Boosting con XGboost:
Se utilizaron las mismas variables que para el modelo anterior y se exploraron distintos hiperparametros:


```
for gamma in list(arange(0,0.21,0.01)):
    for lr in list(arange(0.05,0.65,0.05)):
        for md in list(range(1,10,1)):
            for est in list(range(100,1600,100)):
                best_params = {'gamma': gamma, 'learning_rate': lr, 'max_depth': md, 'n_estimators': est}
                xgbo = xgb.XGBRegressor(**best_params)
                scores = cross_val_score(xgbo, X_train, y_train, cv=10)
                xgb_matrix.append([scores.mean(),gamma,lr,md,est])
                print(scores.mean())
```

De lo anterior se extrajo el modelo con los mejores parametros:

```
best_params = {'gamma': 0.02, 'learning_rate': 0.15, 'max_depth': 3, 'n_estimators': 200}
```

Obteniendo :

```
XGBoost - MSLE = 0.0185
```

Lo que si es una mejora suficiente respecto al anterior dejando este como el modelo final(Todos los detalles de el preprocesamiento, baseline y entrenamiento pueden encontrarse en el archivo [HousePricing.ipynb](https://github.com/TomasCornejo/Predictivo-Precios-de-Casas/blob/master/HousePricing.ipynb) )

Además la implementación quedará en el archivo [modeloPrecioCasa.py](https://github.com/TomasCornejo/Predictivo-Precios-de-Casas/blob/master/modeloPrecioCasa.py)

Dicha implementación es un script de python que toma un dataframe de las columnas en cuestión y genera la predicción en un csv.


