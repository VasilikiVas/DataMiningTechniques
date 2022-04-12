import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats.mstats import spearmanr

def get_raw_data():
    """
    Import raw data and remove "unnamed" variable.
    :return: Dataframe with raw data.
    """

    raw_data = pd.read_csv('dataset_mood_smartphone.csv')
    raw_data = raw_data.drop('Unnamed: 0', axis=1)
    raw_data['time'] = pd.to_datetime(raw_data['time'])

    return raw_data


def show_feature_distribution(data, features, structured_data=False):
    """
    Show distribution of specified attributes.
    :param data: data of features to show distribution from.
    :param features: Set of features to include.
    :param structured_data: Indicates whether data is in structured or unstructured format.
    """

    num = 1
    fig = plt.figure(figsize=(30, 15))
    for var in features:
        ax = fig.add_subplot(5, 4, num)
        if structured_data:
            hist = np.asarray(data[var])
        else:
            hist = list(data.loc[data['variable'] == var].value)
        ax.hist(hist, bins=50)
        ax.set_xlabel('Value')
        ax.set_ylabel('Number of Entries')
        ax.set_title('Distribution of ' + var)
        num += 1
    plt.show()
    plt.clf()


class DataLoader:
    def __init__(self):
        self.raw_data = get_raw_data()
        self.mean_vars = self.raw_data.variable.unique()[0:3]
        self.sum_vars = self.raw_data.variable.unique()[3:]
        self.all_vars = self.raw_data.variable.unique()
        self.dates = pd.date_range(start=self.raw_data.time.min().round('D'),
                                   end=self.raw_data.time.max().round('D'), freq='D')
        self.ids = self.raw_data['id'].unique()
        self.mood_range = [*range(1, 11, 1)]
        self.arousal_valence_range = [*range(-2, 3, 1)]
        self.time_features = ['screen', 'appCat.builtin', 'appCat.communication',
                         'appCat.entertainment', 'appCat.finance', 'appCat.game',
                         'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
                         'appCat.unknown', 'appCat.utilities', 'appCat.weather']

    def remove_invalid_data(self, raw_data):
        """
        Count and remove invalid and NaN values.
        :param raw_data: Raw data without any preproccessing.
        :return: preprocessed data without NaN values or values outside the defined range.
        """

        invalid_values = {i: 0 for i in self.all_vars}
        NaN_values = {i: 0 for i in self.all_vars}

        for index, row in raw_data.iterrows():
            if row['variable'] == "mood":
                if np.isnan(row['value']):
                    NaN_values["mood"] += 1
                elif row['value'] not in self.mood_range:
                    invalid_values["mood"] += 1
                    raw_data.at[index, "value"] = np.nan
            if row['variable'] == "circumplex.arousal" or row['variable'] == "circumplex.valence":
                feature_name = row['variable']
                if np.isnan(row['value']):
                    NaN_values[feature_name] += 1
                elif row['value'] not in self.arousal_valence_range:
                    invalid_values[feature_name] += 1
                    raw_data.at[index, "value"] = np.nan
            if row['variable'] == "activity":
                if np.isnan(row['value']):
                    NaN_values["activity"] += 1
                elif row['value'] > 1 or row['value'] < 0:
                    invalid_values["activity"] += 1
                    raw_data.at[index, "value"] = np.nan
            if row['variable'] == "call" or row['variable'] == "sms":
                feature_name = row['variable']
                if np.isnan(row['value']):
                    NaN_values[feature_name] += 1
                elif row['value'] != 1:
                    invalid_values[feature_name] += 1
                    raw_data.at[index, "value"] = np.nan
            if row['variable'] in self.time_features:
                feature_name = row['variable']
                if np.isnan(row['value']):
                    NaN_values[feature_name] += 1
                elif row['value'] < 0 or row['value'] > 86400:
                    invalid_values[feature_name] += 1
                    raw_data.at[index, "value"] = np.nan

        print(invalid_values)
        print(NaN_values)

        preprocessed_data = raw_data.dropna()

        # Get summary statistics
        # Show number of entries per attribute
        pd.DataFrame(preprocessed_data['variable'].value_counts()).plot.bar(xlabel='Attribute', ylabel='Number of Entries',
                                                                            title="Histogram of Attributes", legend=None,
                                                                            figsize=(10, 7))
        plt.show()
        plt.clf()

        return preprocessed_data

    def remove_outliers(self, data):
        """
        Remove values outside defined quantiles.
        :param data: Data to remove outliers from.
        :return Data without outliers.
        """

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(data.groupby('variable', sort=False).describe())

        for variable_name in self.time_features:
            q_low = data.loc[data['variable'] == variable_name].value.quantile(0.00)
            q_high = data.loc[data['variable'] == variable_name].value.quantile(0.75)
            data = data.drop(data[(data.variable == variable_name) & (
                    (data.value < q_low) | (data.value > q_high))].index)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(data.groupby('variable', sort=False).describe())

        return data

    def get_structured_data(self, data):
        """
        Get data of all variables over time for every user.
        :param data: unstructured data to structure.
        :return: Structured data.
        """

        structured_data = pd.DataFrame(np.nan,
                                       index=pd.MultiIndex.from_product([data.id.unique(), self.dates], names=["ID", "time"]),
                                       columns=self.all_vars)

        for i in self.ids:
            id_used = data[data.id == i]
            id_used.index = id_used.time

            for j in self.mean_vars:
                sub_df = id_used[id_used.variable == j].value.resample('D').mean()
                used_dates = np.array(sub_df.index.strftime('%Y-%m-%d'))
                for k in used_dates:
                    structured_data.loc[i, j].loc[k] = sub_df[k]

            for j in self.sum_vars:
                sub_df = id_used[id_used.variable == j].value.resample('D').sum()
                used_dates = np.array(sub_df.index.strftime('%Y-%m-%d'))
                for k in used_dates:
                    structured_data.loc[i, j].loc[k] = sub_df[k]

        structured_data.to_csv('structured_data.csv')

        return structured_data

    def remove_unreported_periods(self, data):
        """
        Remove days with large amounts of unreported data.
        :param data: Data from which unreported periods need to be removed.
        :return: Dataframe with only periods from user with abundant values.
        """

        data_compact = data.drop(data[(np.isnan(data['screen']) |
                                       np.isnan(data['appCat.builtin']) |
                                       np.isnan(data['appCat.communication']) |
                                       np.isnan(data['appCat.entertainment']) |
                                       np.isnan(data['activity']) |
                                       np.isnan(data['appCat.social']) |
                                       np.isnan(data['appCat.other']) |
                                       np.isnan(data['appCat.office']) |
                                       np.isnan(data['mood']) |
                                       np.isnan(data['circumplex.arousal']) |
                                       np.isnan(data['circumplex.valence']))].index)

        data_compact.to_csv('data_compact.csv')

        return data_compact

    def combine_features(self, data):
        """
        Combine multiple features into one.
        :param data: Data from which features need to be combined.
        :return: Dataframe with combined features.
        """

        nan_values = data.isna().sum()
        #print(nan_values)

        data['appCat.money'] = data[['appCat.office', 'appCat.finance']].sum(axis=1, skipna=True)
        data['appCat.leisure'] = data[['appCat.game', 'appCat.entertainment', 'appCat.social']].sum(axis=1, skipna=True)
        data['appCat.convenience'] = data[['appCat.travel', 'appCat.utilities', 'appCat.weather']].sum(axis=1, skipna=True)
        data['appCat.other'] = data[['appCat.other', 'appCat.unknown']].sum(axis=1, skipna=True)
        column_names = ['appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.social',
                        'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']

        data_new_features = data.drop(columns=column_names, axis=1)

        data_new_features.to_csv('data_new_features.csv')

        nan_values = data_new_features.isna().sum()
        #print(nan_values)

        return data_new_features

    def add_features(self, raw_data, data):
        """
        Add "sleep" and "weekend" features.
        :param raw_data: Data from which amount of sleep is determined.
        :param data: Original dataframe to be used to model.
        :return: Dataframe with added features.
        """

        pass

    def interpolate_values(self, data):
        """
        Interpolate missing values between consecutive days.
        :param data: Data with missing values.
        :return: Dataframe with interpolated values.
        """

        data_interpolated = data.interpolate()
        data_interpolated.to_csv('data_interpolated.csv')

        nan_values = data_interpolated.isna().sum()
        #print(nan_values)

        return data_interpolated

    def feature_importance_analysis(self, data):
        """
        Determine significant features by means of statistical tests.
        :param data: Data from which significant features need to be determined.
        :return: Analysis of feature significance.
        """
        uncorrelated = []

        for attribute in self.all_vars:
            if attribute != 'mood':
                # calculate spearman's correlation
                coef, p = spearmanr(data['mood'], data[attribute])
                #print(f'Spearmans correlation coefficient of {attribute}: %.3f' % coef)
                # interpret the significance
                alpha = 0.1
                if p > alpha:
                    #print(f'Samples are uncorrelated for attribute: {attribute} (fail to reject H0) p=%.3f' % p)
                    uncorrelated.append(attribute)
                else:
                    print(f'Spearmans correlation coefficient of {attribute}: %.3f' % coef)
                    print(f'Samples are correlated for attribute: {attribute} (reject H0) p=%.3f' % p)
        print(uncorrelated)


    def create_train_test_split(self, data, window_size=3):
        """
        Create training and testing split by aggregating features values over certain window size.
        :param data: Data from which instances need to be created for the split.
        :return: Training and Testing split.
        """
        
        print(data.iloc[0, 0])

        for i in self.ids:
            series = data.loc[data.iloc[:, 0] == i]
            for start_row in range(len(series) - window_size + 1):
                window = series[start_row:start_row + window_size]
                print(window)

        # Standardize data


if __name__ == "__main__":
    data_loader = DataLoader()
    data = data_loader.raw_data
    data = data_loader.remove_invalid_data(data)
    data = data_loader.remove_outliers(data)
    show_feature_distribution(data, data_loader.all_vars, False)
    data = data_loader.get_structured_data(data)
    data = data_loader.remove_unreported_periods(data)
    show_feature_distribution(data, data_loader.all_vars, True)
    #data = data_loader.combine_features(data)
    data = data_loader.interpolate_values(data)
    data_loader.feature_importance_analysis(data)

