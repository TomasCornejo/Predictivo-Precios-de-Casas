import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler as SScaler

#Variables Globales de rutas de modelo y dataset

model_path : str = "ModeloFinal/xgboost_precio_casa.model"
dataset_path : str = "ModeloFinal/xgboost_precio_casa.model"


#Funciones de preprocesamiento de Datos

#Variables Continuas
def get_var_con(df):
    df_con = df[['OverallQual','OverallCond','YearBuilt',\
             'TotalBsmtSF','GarageYrBlt','Fireplaces',\
             'GarageArea','FullBath','HalfBath',\
             '2ndFlrSF','GrLivArea','YearRemodAdd',\
             'MasVnrArea'
             ]]
    return df_con
    
#Variables para Cambio por diccionario

#Diccionario de cambios
Variables_dict = [ \
[['Utilities'],{'AllPub':4,'NoSewr':3, 'NoSeWa':2, 'ELO':1}], \
[['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',\
  'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC'] \
 , {'Ex':5,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, np.nan : 0}], \
[['BsmtExposure'] , {'Gd':4,'Av':3, 'Mn':2, 'No':1, np.nan : 0}], \
[['BsmtFinType1','BsmtFinType2'] , {'GLQ':6,'ALQ':5, 'BLQ':4, 'Rec':3,'LwQ':2,'Unf':1,np.nan : 0}], \
[['CentralAir'],{'N':0, 'Y':1 ,np.nan : 0}], \
[['GarageFinish'] , {'Fin':3,'RFn':2, 'Unf':1, np.nan : 0}], \
[['Fence'] , {'GdPrv':4,'MnPrv':3, 'GdWo':2, 'MnWw':1, np.nan : 0}] \
]

def get_var_dicc(df):

    df_dict = df[['Utilities',
    'ExterQual','ExterCond','BsmtQual','BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'HeatingQC',
    'CentralAir',
    'KitchenQual',
    'FireplaceQu',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'PoolQC',
    'Fence']]
    for elemento in Variables_dict:
        for columna in elemento[0]:
            df_dict = reemplazo_dict(df_dict, columna, elemento[1])  
    return df_dict

def reemplazo_dict(df, x, dic):
    df[x] = df[x].map(dic).fillna(df[x])
    return df

def AgregaOneHotEncoding(df, x):
    lista_tipos = tuple(df[x].unique())
    sub_df = pd.DataFrame(lista_tipos, columns=[x])
    dum_df = pd.get_dummies(sub_df, columns = [x], prefix = [x] )
    sub_df = sub_df.join(dum_df)
    sub_df
    
    df_final = df.merge(sub_df, how='left', on=x)
    df_final = df_final.drop(x,1)
    return df_final

def get_var_OHE(df):
    #Variables para OneHotEncoding
    df_OHE = df[['MSZoning',
    'Street',
    'Alley',
    'LotShape',
    'LandContour',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Foundation',
    'Heating',
    'Electrical',
    'Functional',
    'GarageType',
    'PavedDrive',
    'MiscFeature',
    'SaleType',
    'SaleCondition']]
    
    for columna in  list(df_OHE.columns):
        df_OHE = AgregaOneHotEncoding(df_OHE, columna)
    return df_OHE

#Funcion final que junta los dataframes de las distintas variables

def Preprocessing(df):
    
    archivocolumnas = pd.read_table("listadecolumnas.txt", names=['columnas'])
    listadecolumnas = list(archivocolumnas['columnas'].apply(lambda x: x.replace("'","")))

    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['GarageArea'] = df['GarageArea'].fillna(0)
    
        
    Scale = SScaler()
    
    X1 = get_var_con(df)
        
    X2 = get_var_dicc(df)
        
    X3 = get_var_OHE(df)
 
    X = pd.concat([X1,X2,X3], axis=1)
        
    for element in [item for item in listadecolumnas if item not in list(X.columns)]:
        X[element] = 0
        
    X = X[listadecolumnas].values
        
    X_val = Scale.fit_transform(X)
        
    Id = df['Id'].values
        
    X_val = np.nan_to_num(X_val, nan = 0)
        
    return Id , X_val


def main(model_path=model_path, dataset_path=dataset_path):
    
    dataset = pd.read_csv(dataset_path)

    Id, dataset =  Preprocessing(dataset)

    modelo = pickle.load(open(model_path, 'rb'))

    prediccion = modelo.predict(dataset)

    Resultado = pd.DataFrame(list(zip(Id,prediccion)), columns = ['Id','SalePrice'])

    Resultado.to_csv('Output_modelo_precios_casa.csv', index=False)

    print('Proceso Terminado')

if __name__ == '__main__':
    main(model_path, dataset_path)


