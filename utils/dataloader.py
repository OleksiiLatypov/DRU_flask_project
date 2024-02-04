import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class DataLoader(object):

    def fit(self, dataset):
        self.dataset = dataset.copy()

    def fill_with_mode(self, df: pd.DataFrame, categorical_null: list) -> pd.DataFrame:
        for el in categorical_null:
            mode_val = df[el].mode()[0]
            df[el] = df[el].fillna(mode_val)
        return df

    def fill_with_mean(self, df: pd.DataFrame, numerical_null: list) -> pd.DataFrame:
        for el in numerical_null:
            mean_val = df[el].mean()
            # df.fillna({el: mean_val}, inplace=True)
            df[el] = df[el].fillna(mean_val)
        return df

    def standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.columns:
            mean_val = data[column].mean()
            std_val = data[column].std()

            # Standardize the column
            data.loc[:, column] = (data[column] - mean_val) / std_val
        return data

    def load_data(self):

        # Drop column Date
        self.dataset = self.dataset.drop('Date', axis=1)

        # Fill Null (or) NaN (or) Missing Values in categorical data
        categorical_features = list(filter(lambda x: self.dataset[x].dtype == 'object', self.dataset))
        categorical_null = list(filter(lambda x: self.dataset[x].isnull().sum(), self.dataset[categorical_features]))
        self.dataset = self.fill_with_mode(self.dataset, categorical_null)

        # Fill Null (or) NaN (or) Missing Values in numerical data
        numerical_features = list(filter(lambda x: self.dataset[x].dtype != 'object', self.dataset))
        numerical_null = list(filter(lambda x: self.dataset[x].isnull().sum(), self.dataset[numerical_features]))
        self.dataset = self.fill_with_mean(self.dataset, numerical_null)

        # Standardize
        numeric_columns = self.dataset.select_dtypes(include=['float64']).columns
        self.dataset[numeric_columns] = self.standardize_columns(self.dataset[numeric_columns])

        # Label Encoder
        le = LabelEncoder()
        le.fit(self.dataset['RainToday'])
        self.dataset['RainToday'] = le.transform(self.dataset['RainToday'])

        le = LabelEncoder()
        le.fit(self.dataset['WindGustDir'])
        self.dataset['WindGustDir'] = le.transform(self.dataset['WindGustDir'])

        le = LabelEncoder()
        le.fit(self.dataset['WindDir9am'])
        self.dataset['WindDir9am'] = le.transform(self.dataset['WindDir9am'])

        le = LabelEncoder()
        le.fit(self.dataset['WindDir3pm'])
        self.dataset['WindDir3pm'] = le.transform(self.dataset['WindDir3pm'])

        le = LabelEncoder()
        le.fit(self.dataset['Location'])
        self.dataset['Location'] = le.transform(self.dataset['Location'])

        return self.dataset


# data_set = pd.read_csv('/Users/oleksiilatypov/Desktop/DataScience_Fundementals/DRU_flask/data/train.csv')
#
# res = DataLoader()
# res.fit(data_set)
# r = res.load_data()
# # #
# if __name__ == '__main__':
#     print()
#     # pprint(r['Location'].unique())
#     # pprint(r.isnull().sum())
#     # print(data_set['WindGustDir'].unique())
#     # print(r['WindGustDir'].unique())
#     # print(data_set['WindDir9am'].unique())
#     # print(r['WindDir9am'].unique())
#     # print(data_set['WindDir3pm'].unique())
#     # print(r['WindDir3pm'].unique())
#     # print(data_set['Location'].unique())
#     # print(sorted(r['Location'].unique()))
#     print(data_set.head(10))
#     print(r.head(10))
#     # print(r.head(10).isnull().sum())
#     # print(r.head(10).info())
