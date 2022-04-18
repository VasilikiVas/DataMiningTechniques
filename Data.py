import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
from scipy.stats.mstats import spearmanr
from collections import Counter
from datetime import timedelta


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


def feature_importance_analysis(data):
    """
    Determine significant features by means of statistical tests.
    :param data: Data from which significant features need to be determined.
    :return: Analysis of feature significance.
    """
    uncorrelated = []

    for attribute in data.columns:
        if attribute != 'mood':
            # calculate spearman's correlation
            coef, p = spearmanr(data['mood'], data[attribute])
            # print(f'Spearmans correlation coefficient of {attribute}: %.3f' % coef)
            # interpret the significance
            alpha = 0.1
            if p > alpha:
                # print(f'Samples are uncorrelated for attribute: {attribute} (fail to reject H0) p=%.3f' % p)
                uncorrelated.append(attribute)
            else:
                print(f'Spearmans correlation coefficient of {attribute}: %.3f' % coef)
                print(f'Samples are correlated for attribute: {attribute} (reject H0) p=%.3f' % p)
    print(uncorrelated)


class DataLoader:
    def __init__(self):
        self.raw_data = get_raw_data()
        self.mean_vars = self.raw_data.variable.unique()[0:4]
        self.sum_vars = self.raw_data.variable.unique()[4:]
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
        self.active_periods = None
        self.q_lows = np.zeros(len(self.time_features))
        self.q_highs = np.zeros(len(self.time_features))

    def remove_invalid_data(self, raw_data):
        """
        Count and remove invalid and NaN values.
        :param raw_data: Raw data without any preprocessing.
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
        pd.DataFrame(preprocessed_data['variable'].value_counts()).plot.bar(xlabel='Attribute',
                                                                            ylabel='Number of Entries',
                                                                            title="Histogram of Attributes",
                                                                            legend=None,
                                                                            figsize=(10, 7))
        plt.show()
        plt.clf()

        return preprocessed_data

    def get_split(self, data):

        nr_days = list(Counter(self.active_periods.get_level_values(0)).values())
        nr_days_train = [math.ceil(0.8 * number) for number in nr_days]

        start = [self.active_periods[0][1].date()]
        end = []

        id = self.ids[0]
        end_temp = None
        for row in self.active_periods:
            if row[0] != id:
                id = row[0]
                start.append(row[1].date())
                end.append(end_temp)
            else:
                end_temp = row[1].date()

        end.append(end_temp)

        train_data = pd.DataFrame(columns=data.columns)
        test_data = pd.DataFrame(columns=data.columns)
        for id, days_train, start_date, end_date in zip(self.ids, nr_days_train, start, end):
            id_data = data[data['id'] == id]
            train_data = pd.concat([train_data, id_data[(id_data['time'].dt.date >= start_date) & (
                        id_data['time'].dt.date <= start_date + timedelta(days=days_train))]])
            test_data = pd.concat([test_data, id_data[
                (id_data['time'].dt.date >= start_date + timedelta(days=days_train) + timedelta(days=1)) & (
                            id_data['time'].dt.date <= end_date)]])

        return train_data, test_data

    def remove_outliers(self, data, train=True):
        """
        Remove values outside defined quantiles.
        :param data: Data to remove outliers from.
        :return Data without outliers.
        """

        for index, variable_name in enumerate(self.time_features):
            if train:
                self.q_lows[index] = data.loc[data['variable'] == variable_name].value.quantile(0.00)
                self.q_highs[index] = data.loc[data['variable'] == variable_name].value.quantile(0.75)

            data = data.drop(data[(data.variable == variable_name) & (
                    (data.value < self.q_lows[index]) | (data.value > self.q_highs[index]))].index)

        return data

    def get_structured_data(self, data):
        """
        Get data of all variables over time for every user.
        :param data: unstructured data to structure.
        :return: Structured data.
        """

        structured_data = pd.DataFrame(np.nan,
                                       index=pd.MultiIndex.from_product([data.id.unique(), self.dates],
                                                                        names=["id", "time"]),
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

        return structured_data

    def combine_features(self, data):
        """
        Combine multiple features into one.
        :param data: Data from which features need to be combined.
        :return: Dataframe with combined features.
        """

        data['appCat.work'] = data[['appCat.office', 'appCat.finance']].sum(axis=1, skipna=True, min_count=1)
        data['appCat.leisure'] = data[['appCat.game', 'appCat.entertainment']].sum(axis=1, skipna=True, min_count=1)
        data['appCat.utility'] = data[['appCat.travel', 'appCat.utilities', 'appCat.weather']].sum(axis=1, skipna=True,
                                                                                                   min_count=1)
        data['appCat.other'] = data[['appCat.other', 'appCat.unknown']].sum(axis=1, skipna=True, min_count=1)
        column_names = ['appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.travel',
                        'appCat.unknown', 'appCat.utilities', 'appCat.weather']

        data_new_features = data.drop(columns=column_names, axis=1)

        return data_new_features

    def remove_unreported_periods(self, data):
        """
        Remove days with large amounts of unreported data.
        :param data: Data from which unreported periods need to be removed.
        :return: Dataframe with only periods from user with abundant values.
        """

        """
        data_compact = data.drop(data[(np.isnan(data['screen']) |
                                       np.isnan(data['appCat.builtin']) |
                                       np.isnan(data['appCat.communication']) |
                                       np.isnan(data['appCat.entertainment']) |
                                       np.isnan(data['activity']) |
                                       np.isnan(data['appCat.social']))].index)
        """

        data_compact = data.dropna(axis=0, thresh=9)
        self.active_periods = data_compact.index

        return data_compact

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

        for var in self.mean_vars:
            data[var] = data[var].interpolate().ffill().bfill()
        data = data.fillna(0)

        return data

    def window_aggregation(self, data, set='train', window_size=3):
        """
        Aggregate features values over certain window size and create input and output files.
        :param data: Data from which instances need to be created for the split.
        :return: input and output files.
        """

        input = []
        input_temporal = []
        output = []
        output_temporal = []

        for i in self.ids:
            series = data.loc[[i]]
            for start_row in range(len(series) - window_size + 1):
                window = series[start_row:start_row + window_size]
                window = window.agg({'mood': 'mean', 'circumplex.arousal': 'mean', 'circumplex.valence': 'mean',
                                     'activity': 'mean', 'screen': 'sum', 'call': 'sum', 'sms': 'sum',
                                     'appCat.builtin': 'sum', 'appCat.communication': 'sum', 'appCat.other': 'sum',
                                     'appCat.social': 'sum', 'appCat.work': 'sum', 'appCat.leisure': 'sum',
                                     'appCat.utility': 'sum'})

                input.append(window.values.tolist())

            for row in range(len(series) - 1):
                input_temporal.append(series.iloc[[row]].values.tolist()[0])

        for i in self.ids:
            series = data.loc[[i]]
            output += series['mood'][window_size:len(series)].values.tolist()
            output_temporal += series['mood'][1:len(series)].values.tolist()

        # Standardize data

        input = pd.DataFrame(input)
        input.to_csv(set + "_input.csv", index=False, header=False)

        input_temporal = pd.DataFrame(input_temporal)
        input_temporal.to_csv(set + "_input_temporal.csv", index=False, header=False)

        output = pd.DataFrame(output)
        output.to_csv(set + "_output.csv", index=False, header=False)

        output_temporal = pd.DataFrame(output_temporal)
        output_temporal.to_csv(set + "_output_temporal.csv", index=False, header=False)


if __name__ == "__main__":
    # Get raw data
    data_loader = DataLoader()
    raw_data = data_loader.raw_data
    raw_data = data_loader.remove_invalid_data(raw_data)

    # Prepare appropriate active period windows
    preprocessed_data = data_loader.get_structured_data(raw_data)
    preprocessed_data = data_loader.combine_features(preprocessed_data)
    preprocessed_data = data_loader.remove_unreported_periods(preprocessed_data)

    # split data
    train_data, test_data = data_loader.get_split(raw_data)

    # prepare train data
    train_data = data_loader.remove_outliers(train_data, train=True)
    show_feature_distribution(train_data, data_loader.all_vars, False)
    train_data = data_loader.get_structured_data(train_data)
    train_data = data_loader.combine_features(train_data)
    train_data = data_loader.remove_unreported_periods(train_data)
    train_data = data_loader.interpolate_values(train_data)
    data_loader.window_aggregation(train_data, set="train")

    # prepare test data
    test_data = data_loader.remove_outliers(test_data, train=False)
    show_feature_distribution(test_data, data_loader.all_vars, False)
    test_data = data_loader.get_structured_data(test_data)
    test_data = data_loader.combine_features(test_data)
    test_data = data_loader.remove_unreported_periods(test_data)
    test_data = data_loader.interpolate_values(test_data)
    data_loader.window_aggregation(test_data, set="test")

    # Analyze features
    feature_importance_analysis(train_data)
