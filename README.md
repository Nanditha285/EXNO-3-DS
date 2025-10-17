## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:


import pandas as pd

import numpy as np

from scipy import stats

df=pd.read_csv("data.csv")

df

<img width="749" height="565" alt="image" src="https://github.com/user-attachments/assets/f74bdf36-a1d0-4d6e-99e2-bccdae35fe43" />

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

climate=['Cold','Warm','Hot','Very Hot']

ele=OrdinalEncoder(categories=[climate])

ele.fit_transform(df[["Ord_1"]])

<img width="231" height="306" alt="image" src="https://github.com/user-attachments/assets/0c8fe32a-b5d2-479f-a9d1-469a6bf94fc6" />

df['bo2']=ele.fit_transform(df[['Ord_1']])

df
<img width="775" height="582" alt="image" src="https://github.com/user-attachments/assets/8b6b27e2-af6c-4d04-931f-40dd70deba50" />


le=LabelEncoder()

df2=df.copy()

df2['Ord_2']=le.fit_transform(df2['Ord_2'])

df2


<img width="779" height="549" alt="image" src="https://github.com/user-attachments/assets/378b4afe-ec7e-4b76-b822-31f2911979b0" />

from sklearn.preprocessing import OneHotEncoder 

ohe=OneHotEncoder()

df3=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['City']]))

df2=pd.concat([enc,df3],axis=1)

df2

<img width="1298" height="559" alt="image" src="https://github.com/user-attachments/assets/0af9aef0-1ad1-429d-b01f-71b99d5fb994" />


pd.get_dummies(df,columns=['City'])

<img width="1304" height="568" alt="image" src="https://github.com/user-attachments/assets/48c18644-dbcc-48d9-867c-42f78f440b43" />

pip install --upgrade category_encoders

<img width="1244" height="307" alt="image" src="https://github.com/user-attachments/assets/41163327-d4c0-459e-884b-3d0b7e6c780a" />

from category_encoders import BinaryEncoder

from category_encoders import BinaryEncoder

import pandas as pd

df=pd.read_csv("C:\Users\priya\Downloads\data.csv")

be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

df1=pd.concat([df,nd],axis=1)

df1=df.copy()

df1

<img width="736" height="576" alt="image" src="https://github.com/user-attachments/assets/c99da0e3-4054-4ec5-a5f8-d9c4cf1e12fd" />

from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"])

cc=pd.concat([cc,new],axis=1)

cc

<img width="840" height="575" alt="image" src="https://github.com/user-attachments/assets/b157e3d6-72b0-4218-b31c-e04fba1481a5" />

import pandas as pd

from scipy import stats

import numpy as np

df=pd.read_csv("Data_to_Transform.csv")

df

<img width="1121" height="658" alt="image" src="https://github.com/user-attachments/assets/17123b38-708f-40dc-9fce-6e9983d28f9e" />

df.skew()

<img width="510" height="163" alt="image" src="https://github.com/user-attachments/assets/064cf566-da49-4c05-a454-426e24095d1e" />

np.log(df["Highly Positive Skew"])

<img width="724" height="349" alt="image" src="https://github.com/user-attachments/assets/2042e332-5fb3-4b23-9eb8-9fd428da89d7" />

np.reciprocal(df["Highly Positive Skew"])

<img width="738" height="376" alt="image" src="https://github.com/user-attachments/assets/6cd1111e-30ee-4403-8dbd-899364c37804" />

np.reciprocal(df["Moderate Positive Skew"])

<img width="763" height="370" alt="image" src="https://github.com/user-attachments/assets/95521284-2bbc-4576-9d56-c6c53c156e3b" />

np.square(df["Highly Positive Skew"])

<img width="744" height="367" alt="image" src="https://github.com/user-attachments/assets/eb19c6da-5910-4b5a-b840-3c9e35bcad72" />

df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])

df

<img width="1155" height="506" alt="image" src="https://github.com/user-attachments/assets/ec4c00d1-fa89-44d0-bff6-9a802c2273b0" />

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df['Moderate Negative Skew'])

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])

df

<img width="1178" height="447" alt="image" src="https://github.com/user-attachments/assets/ecd50c3c-4bc6-4f8e-a192-a474230848d5" />

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import scipy.stats as stats

sm.qqplot(df['Moderate Negative Skew'],line='45')

plt.show()

<img width="1011" height="750" alt="image" src="https://github.com/user-attachments/assets/14515b6b-14fb-49cc-8253-8f55e1302931" />

sm.qqplot(df['Moderate Negative Skew_1'],line='45')

<img width="973" height="696" alt="image" src="https://github.com/user-attachments/assets/7c9d2870-e4a0-4b25-91ef-93efdfab8535" />

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])

sm.qqplot(df['Highly Negative Skew'],line='45')

plt.show()

<img width="1088" height="732" alt="image" src="https://github.com/user-attachments/assets/b6f13a49-d6e8-45fc-99c6-e7b351549d29" />

sm.qqplot(df['Highly Negative Skew_1'],line='45')

plt.show()

<img width="989" height="700" alt="image" src="https://github.com/user-attachments/assets/eaa02165-5f0c-4858-ab6a-9bb8ea1258c8" />

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')

<img width="996" height="702" alt="image" src="https://github.com/user-attachments/assets/714a193b-97a1-45dd-a523-894688b16c13" />




# RESULT:
       thus  performed Feature Encoding and Transformation process and save the data to a file.


       
