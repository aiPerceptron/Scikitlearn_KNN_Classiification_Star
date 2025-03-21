# dataset link: https://www.kaggle.com/datasets/deepu1109/star-dataset 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("/Users/Username/Downloads/star_spreadsheet.csv")
starTemp = np.array(df["Temperature (K)"]) # cutting the cake slice of the whole star spreadsheet to show temp
# print(starTemp)
plt.plot(((starTemp - 273)*9/5) + 32)
plt.xlabel("Star List")
plt.ylabel("Temperature in Fahrenheit")
plt.title("Star Temperature Graph")
plt.grid()
plt.savefig("StarTempGraph.png")

#df.columns # this command shows the names of the columns for easy copy
df_features = df[['Temperature (K)', 'Luminosity(L/Lo)', 
    'Radius(R/Ro)', 'Absolute magnitude(Mv)', 
    'Star color', 'Spectral Class']]

df_target = df["Star type"]

star_color = np.array(df['Star color'].str.lower()).reshape([-1,1])

spectral_class = np.array(df['Spectral Class'].str.lower()).reshape([-1,1])

y = np.array(df["Star type"])

# choose your transformer

OE = OrdinalEncoder()

# fitting your transformer of choice with the training data

OE.fit(star_color)

# transform all the data, including the testing data

OE_star_color = OE.transform(star_color)

# fitting your transformer of choice with the training data

OE.fit(spectral_class)

# transform all the data, including the testing data

OE_spectral_class = OE.transform(spectral_class)

# print the results

plt.figure()

plt.hist(OE_star_color, bins=13)
plt.grid()
plt.xlabel("Star Color")
# print the results

plt.figure()

plt.hist(OE_spectral_class, bins=13)
plt.grid()
plt.xlabel("Spectral Class")

df["Spectral Class"] = OE_spectral_class
df["Star color"] = OE_star_color
X = np.array(df[['Temperature (K)', 'Luminosity(L/Lo)', 
    'Radius(R/Ro)', 'Absolute magnitude(Mv)', 
    'Star color', 'Spectral Class']])

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

# step two: picking the algorithm
algo = KNeighborsClassifier(1) # 1 is the best for this.

# step three: Doing the training

algo.fit(X_train, y_train)

# step four: the prediction

y_pred = algo.predict(X_test)

# step five: plotting and analysis

accurate_score = round(accuracy_score(y_test, y_pred)*100, 2) 
print(str(accurate_score) + "% accuracy")

temp_train = X_train[:, 0] # ALL of the rows but only the first column
lum_train = X_train[:, 1] # ALL of the rows but only the second column
radius_train = X_train[:, 2] # ALL of the rows but only the third column
mag_train = X_train[:, 3] # ALL of the rows but only the fourth column
color_train = X_train[:, 4]
spectral_train = X_train[:, 5]

temp_test = X_test[:, 0]
lum_test = X_test[:, 1]
radius_test = X_test[:, 2]
mag_test = X_test[:, 3]
color_test = X_test[:, 4]
spectral_test = X_test[:, 5]
