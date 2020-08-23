# Precios de Casas - Predictivo 

Este trabajo es un modelo predictivo para una variable continua usando una base extraida de las competiciones Kaggle.

El dataset está conformado por información de casas donde las variables independientes son areas, cantidades y diferentes clasificaciones, mientras que la variable dependiente es el precio de la misma.

La principal idea es plantear el problema y mostrar como es la forma de trabajo que utilizo para solucionarlo.

## Metodología

Para desarrollar esta solución se seguiran las siguiente metodología:


1er Entrega: Baseline

Modelo creado a partir solo de las variables numéricas que tienen alta correlación con la variable dependiente, a modo Baseline para tener un punto de partida.
Los resultadosen la primera entrega son:

```
Linear        | MSLE = 0.033

Tree          | MSLE = 0.047

RandomForest  | MSLE = 0.025

MultiLayer    | MSLE = 0.045
```
