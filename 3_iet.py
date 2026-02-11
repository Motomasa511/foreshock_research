import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
import geopandas as gpd
from shapely.geometry import Point
from datetime import timedelta
from tqdm import tqdm
from math import radians, sin, cos
import pickle
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import gamma as gm

mainshock_df_old = pd.read_csv(f'mainshock_df_old.csv', parse_dates=["datetime"])
mainshock_df_new = pd.read_csv(f'mainshock_df_new.csv', parse_dates=["datetime"])

df_old = pd.read_csv("df_inland_old.csv", parse_dates=["datetime"])
df_new = pd.read_csv("df_inland_new.csv", parse_dates=["datetime"])

print('df set done')


def find_nearby_events(df, mainshock_df, distance_threshold_km=10.0, day_threshold=380):
    nearby_event_dfs = []

    for _, row in tqdm(mainshock_df.iterrows(), total=len(mainshock_df)):
        main_time = pd.to_datetime(row["datetime"])
        main_lat = row["latitude"]
        main_lon = row["longitude"]
        main_depth = row["depth"]
        mx, my, mz = row["x"], row["y"], row["z"]

        time_mask = (pd.to_datetime(df["datetime"]) >= main_time - timedelta(days=day_threshold)) & \
                    (pd.to_datetime(df["datetime"]) < main_time)

        df_candidate = df[time_mask].copy()
        if df_candidate.empty:
            nearby_event_dfs.append(pd.DataFrame())
            continue

        dx = df_candidate["x"].values - mx
        dy = df_candidate["y"].values - my
        dz = df_candidate["z"].values - mz
        df_candidate["distance"] = np.sqrt(dx*dx + dy*dy + dz*dz)

        df_nearby = df_candidate[df_candidate["distance"] <= distance_threshold_km].copy()

        nearby_event_dfs.append(df_nearby)

    return nearby_event_dfs

def gamma_fit(data):
    mean = np.mean(data)
    N = len(data)
    sum_log = np.sum(np.log(data))

    L_max = -np.inf
    g_max = 0
    g = 1e-3
    while g < 1000:
        L = (g - 1) * sum_log - N * (g + g * np.log(mean / g) + gammaln(g))
        if L > L_max:
            g_max = g
            L_max = L
        g += 1e-3

    mu = g_max / mean
    return g_max, mu

def estimate_background_parameters(mainshock_df, nearby_event_dfs, min_days=28, max_days=388):
    
    mu_list = []
    gamma_list = []

    for mainshock, df_nearby in tqdm(zip(mainshock_df.itertuples(), nearby_event_dfs), 
                                     total=len(mainshock_df)):

        main_time = mainshock.datetime

        mask = (df_nearby["datetime"] >= main_time - timedelta(days=max_days)) & \
               (df_nearby["datetime"] <= main_time - timedelta(days=min_days))
        df_window = df_nearby[mask].copy()

        # inter-event times
        df_window = df_window.sort_values("datetime")
        times = df_window["datetime"].values

        if len(times) == 0:
            inter_event_times = np.array([1e3])
        elif len(times) == 1:
            inter_event_times = np.array([360.0])
        else:
            inter_event_times = np.diff(times).astype("timedelta64[s]").astype(float) / (3600 * 24)
            T_days = 360
            iet_n1 = ((times[0] + np.timedelta64(T_days, 'D')) - times[-1]).astype("timedelta64[s]").astype(float) / (3600 * 24)
            inter_event_times = np.append(inter_event_times, iet_n1)
            inter_event_times = inter_event_times[(inter_event_times > 0) & ~np.isnan(inter_event_times)]

        # ガンマ分布フィッティング
        gamma, mu = gamma_fit(inter_event_times)

        gamma_list.append(gamma)
        mu_list.append(mu)

    # mainshock_df に追加して返す
    mainshock_df = mainshock_df.copy()
    mainshock_df["background_mu"] = mu_list
    mainshock_df["background_gamma"] = gamma_list

    return mainshock_df

def estimate_gamma_moments(data):
    
    mean = np.mean(data)
    var = np.var(data, ddof=1)

    shape = mean ** 2 / var
    scale = var / mean
    return shape, 1/scale

def simulate_event_counts_within_T_days(mainshock_df, nearby_event_dfs, T_days=28, n_trials=50000):

    foreshock_99 = []
    foreshock_rate = []
    mu_list = []
    gamma_list = []

    for idx, (_, row) in enumerate(tqdm(mainshock_df.iterrows(), total=len(mainshock_df))):
        main_time = row['datetime']
        df_nearby = nearby_event_dfs[idx]
        mask = (df_nearby["datetime"] >= main_time - timedelta(days=T_days)) & \
               (df_nearby["datetime"] < main_time)
        N_fore = mask.sum()

        mu = row["background_mu"]
        gamma_shape = row["background_gamma"]
        scale = 1 / mu
        max_len = max(200, int(T_days * mu * 1.5 / gamma_shape))

        samples = gm.rvs(a=gamma_shape, scale=scale, size=(n_trials, max_len))
        cumsum = np.cumsum(samples, axis=1)
        N = (cumsum < T_days).sum(axis=1)
        N_positive = N[N > 0]

        if len(N_positive) == 0:
            mu_list.append(np.nan)
            gamma_list.append(np.nan)
            if N_fore == 0:
                rate = 1
            else:
                rate = 1e-5
        else:
            gamma_est, mu_est = estimate_gamma_moments(N)
            mu_list.append(mu_est)
            gamma_list.append(gamma_est)
            rate = 1-gm.cdf(N_fore, a=gamma_est, scale=1 / mu_est)
        foreshock_rate.append(rate)
        foreshock_99.append(rate<0.01)

    mainshock_df = mainshock_df.copy()
    mainshock_df["iet_mu"] = mu_list
    mainshock_df["iet_gamma"] = gamma_list
    mainshock_df["iet_rate"] = foreshock_rate
    mainshock_df["TF_iet"] = foreshock_99

    return mainshock_df

if "TF_iet" in mainshock_df_old.columns:
    print("already calculated, skip")
else:
    nearby_event_dfs_old = find_nearby_events(df_old, mainshock_df_old, distance_threshold_km=10.0, day_threshold=380)
    nearby_event_dfs_new = find_nearby_events(df_new, mainshock_df_new, distance_threshold_km=10.0, day_threshold=380)
    mainshock_df_old = estimate_background_parameters(mainshock_df_old, nearby_event_dfs_old)
    mainshock_df_new = estimate_background_parameters(mainshock_df_new, nearby_event_dfs_new)
    mainshock_df_old = simulate_event_counts_within_T_days(mainshock_df_old, nearby_event_dfs_old)
    mainshock_df_new = simulate_event_counts_within_T_days(mainshock_df_new, nearby_event_dfs_new)

    mainshock_df_old.to_csv(f"mainshock_df_old.csv", index=False)
    mainshock_df_new.to_csv(f"mainshock_df_new.csv", index=False)

# foeeshock occurrence rate
print("oreshock occurrence rate : ")

def occurrence_rate():
    df_old = pd.read_csv("mainshock_df_old.csv")
    df_new = pd.read_csv("mainshock_df_new.csv")
    foreshock_list_old = df_old["TF_iet"]
    foreshock_list_new = df_new["TF_iet"]

    print(f"old : \n{np.sum(foreshock_list_old)}/{len(foreshock_list_old)} = {100*np.sum(foreshock_list_old)/len(foreshock_list_old):.3f}%\n")
    print(f"new : \n{np.sum(foreshock_list_new)}/{len(foreshock_list_new)} = {100*np.sum(foreshock_list_new)/len(foreshock_list_new):.3f}%\n")
    print(f"old+new : \n({np.sum(foreshock_list_old)}+{np.sum(foreshock_list_new)})/({len(foreshock_list_old)}+{len(foreshock_list_new)}) = {100*(np.sum(foreshock_list_old)+np.sum(foreshock_list_new))/(len(foreshock_list_old)+len(foreshock_list_new)):.3f}%")

occurrence_rate()








