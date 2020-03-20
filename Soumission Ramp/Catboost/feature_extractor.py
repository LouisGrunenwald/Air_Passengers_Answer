import pandas as pd
import numpy as np
import os


def _haversine_distance(lat_A, lon_A, lat_B, lon_B):
    earth_radius = 6373
    distance = np.sin((lat_B - lat_A) / 2) ** 2 + np.cos(lat_A) * \
            np.cos(lat_B) * np.sin((lon_B - lon_A) / 2) ** 2
    distance = 2 * earth_radius * np.arcsin(distance)
    return distance


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df.copy()
        path = os.path.dirname(__file__)
        final = pd.read_csv(os.path.join(path, 'external_data.csv'))
        final["Date"]=pd.to_datetime(final["Date"])
        final["Events"]=final["Events"].fillna("Sunny")
        final2 = final.join(pd.get_dummies(final['Events']))
        

        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date -pd.to_datetime("1970-01-01")).days)
        X_df_Depart = X_encoded.drop("Arrival", axis=1)
        X_df_Arrivee = X_encoded.drop("Departure", axis=1)

        fusion_depart = X_df_Depart.merge(final2.add_suffix('_depart'),how="left", \
        left_on = ["DateOfDeparture","Departure"],right_on=["Date_depart","AirPort_depart"])
        fusion_depart = fusion_depart.drop('Events_depart', axis=1)
        fusion_arrivee = X_df_Arrivee.merge(final2.add_suffix('_arrivee'),how="left",\
        left_on = ["DateOfDeparture","Arrival"],right_on=["Date_arrivee","AirPort_arrivee"])
        fusion_arrivee = fusion_arrivee.drop('Events_arrivee', axis=1)
        fusion_arrivee = fusion_arrivee.drop(["DateOfDeparture","WeeksToDeparture",\
                                                    "std_wtd","year","month","day","weekday","week","n_days"],axis=1)
        perfect = fusion_depart.merge(fusion_arrivee, left_index=True, right_index=True)

        perfect["Precipitationmm_arrivee"]=np.where(perfect["Precipitationmm_arrivee"] == 'T',0,perfect["Precipitationmm_arrivee"])
        perfect["Precipitationmm_depart"]=np.where(perfect["Precipitationmm_depart"] == 'T',0,perfect["Precipitationmm_depart"])
        perfect = perfect.join(pd.get_dummies(perfect['Departure'], prefix='depart'))
        perfect = perfect.join(pd.get_dummies(perfect['Arrival'], prefix='arrivee'))
        perfect = perfect.drop('Departure', axis=1)
        perfect = perfect.drop('Arrival', axis=1)
        liste_flo = ["lat_arrivee","lon_arrivee","Precipitationmm_arrivee","lat_depart","lon_depart","Precipitationmm_depart"]
        perfect[liste_flo]=perfect[liste_flo].astype(float)
        perfect['distance'] = _haversine_distance(\
                perfect['lat_arrivee'].values,\
                perfect['lon_arrivee'].values,\
                perfect['lat_depart'].values,\
                perfect['lon_depart'].values)
        perfect=perfect.drop(["DateOfDeparture_arrivee","Date_depart",\
                    "DateOfDeparture_depart","Date_arrivee","City_arrivee",\
                    "State_arrivee","City_depart","State_depart","AirPort_arrivee",\
                    "AirPort_depart","is_weekend_depart","is_holiday_depart",\
                    "is_close_from_weekend_or_holiday_depart"],axis=1)
        perfect = perfect.rename(columns={'is_weekend_arrivee': 'is_weekend',\
                    'is_holiday_arrivee':'is_holiday',\
                    'is_close_from_weekend_or_holiday_arrivee':'is_close_from_weekend_or_holiday'})
        for i in [c for c, d in zip (perfect.columns,perfect.dtypes) if d==np.bool] :
            perfect[i]=np.where(perfect[i] == False,0,1)   
        perfect.columns=perfect.columns.str.replace("-","_",regex=True)
        del perfect["DateOfDeparture"]

        X_array = perfect.values
        return X_array