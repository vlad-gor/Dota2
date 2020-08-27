import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


#1. Читаем файл с данными
df = pd.read_csv("features.csv", index_col="match_id")

#2. Удаляем лишние признаки
df.drop(['duration',
         'tower_status_radiant',
         'tower_status_dire',
         'barracks_status_radiant',
         'barracks_status_dire'],
          axis=1, inplace=True)

#3. Находим пропуски в данных
lost=len(df)-df.count()
lost = lost[lost>0].sort_values(ascending=False) / len(df)
print(lost)
'''
first_blood_player2            0.452402
radiant_flying_courier_time    0.282619
dire_flying_courier_time       0.268415
first_blood_player1            0.201100
first_blood_team               0.201100
first_blood_time               0.201100
dire_bottle_time               0.166029
radiant_bottle_time            0.161380
radiant_first_ward_time        0.018883
dire_first_ward_time           0.018780
radiant_courier_time           0.007117
dire_courier_time              0.006953
'''
'''
first_blood_time - в 20% случаев первые 5 минут игры нет столкновений и кровь не проливается
аналогичный показатель для first_blood_team

самые редкие явления - приобретение командами =courier= в начале игры
radiant_courier_time < 1%
dire_courier_time    < 1%
'''

#4. Заменяем пропуски на нули 
df.fillna(0, inplace=True)

#5. Целевая переменная 'radiant_win', отделяем ее от данных
X_train = df.drop('radiant_win',axis=1)
y_train = df['radiant_win']

#6. Обучаем градиентный бустинг

cross_val = KFold(n_splits=5, shuffle=True, random_state=37)  

def score_gb(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}

    for n_estimators in [10, 20, 30, 50, 100, 250]:
        print(f"n_estimators={n_estimators}")
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)

        start_time = datetime.datetime.now()
        score = cross_val_score(model, X, y, cv=cross_val, scoring="roc_auc", n_jobs=-1).mean()
        print(f"Score: {score:.4f}")
        print(f"Time elapsed: {datetime.datetime.now() - start_time}")

        scores[n_estimators] = score
        print()
        
    return pd.Series(scores)

# scores = score_gb(X_train, y_train)
# scores.plot()

# plt.show()

'''
n_estimators=10
Score: 0.6654
Time elapsed: 0:00:27.537906

n_estimators=20
Score: 0.6822
Time elapsed: 0:00:54.306444

 

n_estimators=50
Score: 0.6975
Time elapsed: 0:02:01.636231

n_estimators=100
Score: 0.7065
Time elapsed: 0:04:02.484176

n_estimators=250
Score: 0.7163
Time elapsed: 0:09:51.723190
'''

#####################################################################################

#7. Обучаем логистическую регрессию
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

def score_lr(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}

    for i in range(-5, 6):
        C = 10.0 ** i

        print(f"C={C}")
        model = LogisticRegression(C=C, random_state=42)

        start_time = datetime.datetime.now()
        score = cross_val_score(model, X, y, cv=cross_val, scoring="roc_auc", n_jobs=-1).mean()
        print(f"Score: {score:.3f}")
        print(f"Time elapsed: {datetime.datetime.now() - start_time}")

        scores[i] = score
        print()

    return pd.Series(scores)

# scores = score_lr(X_train, y_train)
# scores.plot()

# plt.show()

'''
C=1e-05
Score: 0.695
Time elapsed: 0:00:04.272109

C=0.0001
Score: 0.711
Time elapsed: 0:00:05.807497

C=0.001
Score: 0.716
Time elapsed: 0:00:09.456401

C=0.01
Score: 0.717
Time elapsed: 0:00:11.909058

C=0.1
Score: 0.717
Time elapsed: 0:00:12.328881

C=1.0
Score: 0.717
Time elapsed: 0:00:12.555295

C=10.0
Score: 0.717
Time elapsed: 0:00:11.833938

C=100.0
Score: 0.717
Time elapsed: 0:00:11.966060

C=1000.0
Score: 0.717
Time elapsed: 0:00:11.851004

C=10000.0
Score: 0.717
Time elapsed: 0:00:11.758836

C=100000.0
Score: 0.717
Time elapsed: 0:00:12.072340
'''

# Наилучшее значение показателя AUC-ROC достигается при С = 0.01 и равно 0.717

# 8. Убираем из выборки категориальные признаки
cat_features=['lobby_type',
              'r1_hero', 'r2_hero','r3_hero','r4_hero','r5_hero',
              'd1_hero', 'd2_hero','d3_hero','d4_hero','d5_hero']

X_train.drop(cat_features, axis=1, inplace=True) 

# scores = score_lr(X_train, y_train)
# scores.plot()

# plt.show()

'''
C=1e-05
Score: 0.695
Time elapsed: 0:00:04.117678

C=0.0001
Score: 0.711
Time elapsed: 0:00:05.521658

C=0.001
Score: 0.716
Time elapsed: 0:00:08.739467

C=0.01
Score: 0.717
Time elapsed: 0:00:10.953591

C=0.1
Score: 0.717
Time elapsed: 0:00:11.433563

C=1.0
Score: 0.717
Time elapsed: 0:00:11.479339

C=10.0
Score: 0.717
Time elapsed: 0:00:11.473897

C=100.0
Score: 0.717
Time elapsed: 0:00:11.454752

C=1000.0
Score: 0.717
Time elapsed: 0:00:11.500613

C=10000.0
Score: 0.717
Time elapsed: 0:00:11.530505

C=100000.0
Score: 0.717
Time elapsed: 0:00:12.503361
'''

# Удаление категоральных признаков не повлияло на качество предсказания

# 9. Выясним количество уникальных героев в игре
unique_heroes = np.unique(
    df[['r1_hero', 'r2_hero','r3_hero','r4_hero','r5_hero',
        'd1_hero', 'd2_hero','d3_hero','d4_hero','d5_hero']].values.ravel())
N = max(unique_heroes)
print(unique_heroes)
print(len(unique_heroes))
print(N)

# Число уникальных героев 108, максимальный ID героя - 112

# 10. Воспользуемся подходом мешок слов

# Создается N признаков. Если герой играл за команду radiant, признак равен 1, если за команду dire, то -1.
# Если не участвовал в игре, то 0

def get_pick(data: pd.DataFrame) -> pd.DataFrame:
    X_pick = np.zeros((data.shape[0], N))

    for i, match_id in enumerate(data.index):
        for p in range(1, 6):
            X_pick[i, data.loc[match_id, f"r{p}_hero"] - 1] = 1
            X_pick[i, data.loc[match_id, f"d{p}_hero"] - 1] = -1

    return pd.DataFrame(X_pick, index=data.index, columns=[f"hero_{i}" for i in range(N)])

X_pick = get_pick(df)
print(X_pick.head())

X_train = pd.concat([X_train, X_pick], axis=1)

# scores = score_lr(X_train, y_train)
# scores.plot()

# plt.show()

'''
C=1e-05
Score: 0.699
Time elapsed: 0:00:05.823848

C=0.0001
Score: 0.725
Time elapsed: 0:00:07.606816

C=0.001
Score: 0.746
Time elapsed: 0:00:12.631312

C=0.01
Score: 0.752
Time elapsed: 0:00:18.430193

C=0.1
Score: 0.752
Time elapsed: 0:00:24.613963

C=1.0
Score: 0.752
Time elapsed: 0:00:25.748296

C=10.0
Score: 0.752
Time elapsed: 0:00:25.701794

C=100.0
Score: 0.752
Time elapsed: 0:00:25.811116

C=1000.0
Score: 0.752
Time elapsed: 0:00:25.748250

C=10000.0
Score: 0.752
Time elapsed: 0:00:25.706425

C=100000.0
Score: 0.752
Time elapsed: 0:00:25.896542
'''

# Наилучшее значение показателя AUC-ROC достигается при C = 0.1 и равно 0.752
# после преобразования категориальных признаков качество значительно улучшилось

# 11. Построим предсказание вероятности победы для тестовой выборки
model = LogisticRegression(C=0.1, random_state=42)
model.fit(X_train, y_train)

test = pd.read_csv("features_test.csv", index_col="match_id")
test.fillna(0, inplace=True)

X_test = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
X_test.drop(cat_features, axis=1, inplace=True)
X_test = pd.concat([X_test, get_pick(test)], axis=1)

print(X_test.head())

preds = pd.Series(model.predict_proba(X_test)[:, 1])
print(preds.describe())

preds.plot.hist(bins=30)

plt.show()

# По графику видно что предсказанные вероятности соответствуют реальности
# находятся на отрезке [0, 1], не совпадают между собой