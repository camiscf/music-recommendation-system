from os import link
import numpy as np
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pandas as pd

# Autenticação

cid = '6eb385abeb61467ba564419872185201'
secret = '1687cec7ce7f4b2fa357dd7cdc4a15e2'

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#acoustic, bossanova, heavy-metal, electronic, forro, funk, indie-pop, indie, k-pop, pop, clássico, MPB, Samba, Pagode, rock
acoustic = '37i9dQZF1DWXRvPx3nttRN'
bossanova = '37i9dQZF1DX4AyFl3yqHeK'
heavy_metal = '37i9dQZF1DX9qNs32fujYe'
electronic = '37i9dQZF1EQp9BVPsNVof1'
forro = '37i9dQZF1DWZMthVIHWet5'
funk = '37i9dQZF1DWTkIwO2HDifB'
indie_pop = '37i9dQZF1DWWEcRhUVtL8n'
indie = '37i9dQZF1EQqkOPvHGajmW'
k_pop = '37i9dQZF1DX9tPFwDMOaN1'
pop = '37i9dQZF1DX6aTaZa0K6VA'
classico = '37i9dQZF1DWWEJlAGA9gs0'
mpb = '37i9dQZF1DWZ8wKkpiPwjx'
samba = '37i9dQZF1DWTUHOvJwQIMp'
pagode = '37i9dQZF1DX7CZIZFUsRDm'
rock = '37i9dQZF1EQpj7X7UK8OOF'

columns = ["id", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "key", "mode", "url"]
playlists = [acoustic, bossanova, heavy_metal, electronic, forro, funk, indie_pop, indie, k_pop, pop, classico, mpb, samba, pagode, rock]
pd.set_option('display.max_colwidth', None)

# LEITURA DO CSV
db = pd.read_csv('sysMusic\data\out.csv')
# print(df.head()) -> ele leu o csv perfeito
db_copy = db.copy()
# print(df_copy.head())
db_onehot = pd.get_dummies(db_copy, columns=['key', 'mode'])
# print(db_onehot.head())

# ENTRADA
def getId(url):
    # definir se o link entrada começa com https
  if url[0] == 'h':
    return url[31:53]
  else: # começa direto no open.spotify ...
    return url[23:45]

def getInputDF(url):
  inputID = getId(url)
  inputInfo = sp.audio_features(inputID)
  info_list = []
  info_list.extend([inputInfo[0]['id'], inputInfo[0]["danceability"], inputInfo[0]["energy"], inputInfo[0]["loudness"],
                            inputInfo[0]["speechiness"], inputInfo[0]["acousticness"], inputInfo[0]["instrumentalness"],
                            inputInfo[0]["liveness"], inputInfo[0]["valence"], inputInfo[0]["tempo"], inputInfo[0]["key"], inputInfo[0]["mode"]]) 
  info_list.append(url)

  inputTrackName = sp.track(inputID)['name']

  # fazer o one_hot de acordo com a tabela de banco de dados
  new_row = pd.DataFrame([info_list], index=[inputTrackName], columns=columns)
  temp_db = db_copy.append(new_row)
  temp_df = pd.get_dummies(temp_db, columns=['key', 'mode'])
  input_df = temp_df.iloc[[-1]]
  #db_copy.drop([inputTrackName], axis=0)
  return input_df, new_row

def df_to_np(df):
  return df.to_numpy()

def clean_df(one_hot, df):
  '''
    Recebe um dataframe e um boolenao indicando se apliquei one_hot nos dados categoricos (key e mode)
    retorna um novo df pronto para virar numpy array
  '''
  if one_hot == False: # significa que preciso deltar as colunas com informações categóricas
    # checa se tenho a coluna unnamed: 0 que acontece quando tenho upload de arquivos
    if 'Unnamed: 0' in df.columns:
      return df.drop(['Unnamed: 0', 'id', 'key', 'mode', 'url'], axis=1)
    
    return df.drop(['id', 'key', 'mode', 'url'], axis=1)
  
  # no de aplicar o one_hot nos dados categoricos não preciso tira-los do dataframe
  # checa se tenho a coluna unnamed: 0 que acontece quando tenho upload de arquivos
  if 'Unnamed: 0' in df.columns:
    return df.drop(['Unnamed: 0', 'id', 'url'], axis=1)
  
  return df.drop(['id', 'url'], axis=1)

def check_existence(df, input_df):
  '''
    Checo no meu universo (db) se tem a música que o usuário entrou
    Caso tenha, retiro ela do universo
  '''
  db_index = db.index
  condition = input_df["id"].item() == db["id"]
  if db_index[condition].any():
    track_name = db_index[condition].tolist()[0]
    position = db_index.tolist().index(track_name)
    if type(df.index.tolist()[position]) == str:
      temp = df.drop([track_name], axis=0)
      return temp
    else:
      temp = df.drop([position], axis=0)

      return temp

  return df

# METODOS PARA CALCULAR

def calcDistance(x, y):

  return np.linalg.norm(x - y)

def getBestByDistance(matrix, vector):

  qtd_musicas = np.shape(matrix)[0]
  dist = calcDistance(matrix[0], vector)
  rec_id = 0 # id da música recomendada
    
  for i in range(1, qtd_musicas):
    candidate = calcDistance(matrix[i], vector)
    if  candidate < dist: # atualiza a distância
      dist = candidate
      rec_id = i
  
  return rec_id

def getRecommendationByDIstance(input_df, df):

  matrix = df_to_np(df)
  vector = df_to_np(input_df)
  rec_id = getBestByDistance(matrix, vector)
  

  return db.iloc[[rec_id]]

def distance_onehot(input_df):
  # checa se a música entrada não está no df entrado
  db_unique = check_existence(db_onehot.copy(), input_df)
  res = getRecommendationByDIstance(clean_df(True, input_df), clean_df(True, db_unique))

  return res
  
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendation_by_cosine(np_db, np_input):
  '''
    Recebe o database e o input em formato de numpy array
  '''
  cos_sim = cosine_similarity(np_db, np_input)
  val = np.amax(cos_sim)
  rec_id = np.where(cos_sim == val)[0]
  
  # print("Similaridade dos cosenos foi de:", val)
  return db.iloc[rec_id]

def normalize(scaler, df):
  '''
    Recebe o scaler o dataframe para normalizar e retorna o dataframe normalizado
  '''
  df_normalized = scaler.fit_transform(df)
  return pd.DataFrame(df_normalized)

# normaliza o dado
scaler = StandardScaler()
normalized = normalize(scaler, clean_df(True, db_onehot.copy())) # será uma variável Global

def cosine_onehot(input_df):
  
  input_normalized = scaler.transform(clean_df(True, input_df))
  input_normalized = pd.DataFrame(input_normalized)

  df_normalized = check_existence(normalized.copy(), input_df)

  # passando dataframe para numpy
  np_input = df_to_np(input_normalized)
  np_db = df_to_np(df_normalized)

  return get_recommendation_by_cosine(np_db, np_input)



def main(inputs):
  input_df = getInputDF(inputs)[0]
  temp = distance_onehot(input_df)
  getTemp = temp.iloc[[0]].to_dict('records')
  getFinal = getTemp[0]
  # getMusicDistance pega só o nome da música
  getMusicDistance = getFinal["Unnamed: 0"]

  # tentando pegar o hyperlink
  # link_teste pega a url
  link_teste = getFinal["url"]

  # COSSENO MUSICA RECOMENDAÇÃO
  temp_cossine = cosine_onehot(input_df)
  getTemp_cossine = temp_cossine.iloc[[0]].to_dict('records')
  getFinal_cossine = getTemp_cossine[0]
  getMusicCossine = getFinal_cossine["Unnamed: 0"]
  link_teste_cossine = getFinal_cossine["url"]


  return link_teste, getMusicDistance,link_teste_cossine,getMusicCossine
  # return getFinal
  


  