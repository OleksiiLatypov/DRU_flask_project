import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    @staticmethod
    def fill_with_mode(df, categorical_null):
        for el in categorical_null:
            mode_val = df[el].mode()[0]
            df[el].fillna(mode_val, inplace=True)
        return df

    @staticmethod
    def fill_with_mean(df, numerical_null):
        means = []
        stds = []
        factor = 1.5
        d = {}
        for el in numerical_null:
            mean_val = df[el].mean()
            std_val = df[el].std()
            upper_limit = mean_val + std_val * factor
            lower_limit = mean_val - std_val * factor
            d[el] = (upper_limit, lower_limit)
            # df[el] = df[el].clip(lower=lower_limit, upper=upper_limit)
            df[el] = df[el].apply(lambda x: upper_limit if x > upper_limit else (lower_limit if x < lower_limit else x))
        # df.head()
        for column in numerical_null:
            mean_val = df[column].mean()
            df[column] = df[column].fillna(mean_val)
        return df

    @staticmethod
    def encode_data(df, feature_name):

        '''

         function which takes feature name as a parameter and return mapping dictionary to replace(or map) categorical data
         to numerical data.

        '''

        mapping_dict = {}
        unique_values = list(df[feature_name].unique())
        for idx in range(len(unique_values)):
            mapping_dict[unique_values[idx]] = idx
        #print(mapping_dict)
        return mapping_dict

    def load_data(self):
        #df = self.dataset

        # transform data
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        self.dataset['year'] = self.dataset['Date'].dt.year
        self.dataset['month'] = self.dataset['Date'].dt.month
        self.dataset['day'] = self.dataset['Date'].dt.day

        # drop column Date
        self.dataset = self.dataset.drop('Date', axis=1)

        # fill Null (or) NaN (or) Missing Values in data
        categorical_features = list(filter(lambda x: self.dataset[x].dtype == 'object', self.dataset))
        categorical_null = list(filter(lambda x: self.dataset[x].isnull().sum(), self.dataset[categorical_features]))
        #print(categorical_null)
        self.dataset = self.fill_with_mode(self.dataset, categorical_null)
        #print(df[categorical_null].isnull().sum())

        # fill Null (or) NaN (or) Missing Values in numerical data
        numerical_features = list(filter(lambda x: self.dataset[x].dtype != 'object', self.dataset))
        numerical_null = list(filter(lambda x: self.dataset[x].isnull().sum(), self.dataset[numerical_features]))
        self.dataset = self.fill_with_mean(self.dataset, numerical_null)
        #print(df[numerical_null].isnull().sum())
        #print('SHAPE', df.shape)
        # Scaling
        numeric_columns = self.dataset.select_dtypes(include=['float64']).columns
        scaler = StandardScaler()
        self.dataset[numeric_columns] = scaler.fit_transform(self.dataset[numeric_columns])


        self.dataset['RainToday'] = self.dataset['RainToday'].replace({'No': 0, 'Yes': 1})

        # pd.get_dummies(rain['RainToday'],drop_first = True)

        self.dataset['RainTomorrow'] = self.dataset['RainTomorrow'].replace({'No': 0, 'Yes': 1})

        # Encode data
        self.dataset['WindGustDir'] = self.dataset['WindGustDir'].replace(self.encode_data(self.dataset, 'WindGustDir'))
        self.dataset['WindDir9am'] = self.dataset['WindDir9am'].replace(self.encode_data(self.dataset, 'WindDir9am'))
        self.dataset['WindDir3pm'] = self.dataset['WindDir3pm'].replace(self.encode_data(self.dataset, 'WindDir3pm'))
        self.dataset['Location'] = self.dataset['Location'].replace(self.encode_data(self.dataset, 'Location'))

        #print(df.head(10))
        #print(df.isnull().sum())
        return self.dataset


df = pd.read_csv('/Users/oleksiilatypov/Desktop/DataScience_Fundementals/DRU_flask/data/train.csv')

res = DataLoader()
res.fit(df)
# res.load_data()
r = res.load_data()

if __name__ == '__main__':
    print(r.head())
