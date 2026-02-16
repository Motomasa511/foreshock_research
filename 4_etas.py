# for 本震:
    # 本震周りの全期間の地震をdf_nearbyに
    # work.etasとetas.openを作る
    # パラメータフィット(etas.f)しmainshock_dfに追加
    # 20daysでのetasでの地震数を求め、Nobsがポアソン分布のxx%にあるxxを求める
    # xx <0.01 で T or F

# df_nearby_list を保存
# rate を求める

# f2py -c -m etas_module etas_forpy.f


import pandas as pd
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import pickle
import csv
from scipy.stats import poisson
import etas_module

mainshock_df_old = pd.read_csv(f'mainshock_df_old.csv', parse_dates=["datetime"])
mainshock_df_new = pd.read_csv(f'mainshock_df_new.csv', parse_dates=["datetime"])

df_old = pd.read_csv("df_inland_old.csv", parse_dates=["datetime"])
df_new = pd.read_csv("df_inland_new.csv", parse_dates=["datetime"])

print('df set done')

os.makedirs("4_etas", exist_ok=True)
open("4_etas/etas.log", "w").close()

def find_nearby_events(df, mainshock_df, distance_threshold_km=10.0, flag=""):
    nearby_event_dfs = []
    counts = []

    for _, row in tqdm(mainshock_df.iterrows(), total=len(mainshock_df)):
        main_time = pd.to_datetime(row["datetime"])
        main_lat = row["latitude"]
        main_lon = row["longitude"]
        main_depth = row["depth"]
        mx, my, mz = row["x"], row["y"], row["z"]

        df_candidate = df.copy()
        if df_candidate.empty:
            nearby_event_dfs.append(pd.DataFrame())
            continue

        dx = df_candidate["x"].values - mx
        dy = df_candidate["y"].values - my
        dz = df_candidate["z"].values - mz
        df_candidate["distance"] = np.sqrt(dx*dx + dy*dy + dz*dz)

        df_nearby = df_candidate[df_candidate["distance"] <= distance_threshold_km].copy()

        nearby_event_dfs.append(df_nearby)
        counts.append(len(df_nearby))
    
    print(f"nearby events number \nmin : {np.min(counts)} \nmax : {np.max(counts)}")
    print(counts[0])

    with open(f"4_etas/nearby_list4_{flag}.pkl", "wb") as f:
        pickle.dump(nearby_event_dfs, f)

if os.path.exists("4_etas/nearby_list4_old.pkl"):
    print('skip building nearby.pkl')
else:
    find_nearby_events(df_old, mainshock_df_old, flag="old")
    find_nearby_events(df_new, mainshock_df_new, flag="new")
    print("building nearby.pkl done")

def read_nearby_events(i, flag=""):
    with open(f"4_etas/nearby_list4_{flag}.pkl", "rb") as f:
        nearby_event_list4 = pickle.load(f)
    return nearby_event_list4[i]

def make_work_etas(df, t0):
    # t0 : start time (old : 1998-01-01; new : 2016-04-01)

    df = df.dropna(subset=["datetime", "longitude", "latitude", "magnitude", "depth"])
    df = df.sort_values("datetime").reset_index(drop=True)

    t0 = pd.to_datetime(t0)

    z_days = (df["datetime"] - t0).dt.total_seconds() / (24 * 3600)


    years = df["datetime"].dt.year.values
    months = df["datetime"].dt.month.values
    days = df["datetime"].dt.day.values

    with open("4_etas/work.etas", "w") as f:
        f.write("formatted_for_etas\n")

        for i in range(len(df)):
            line = (
                f"{i+1:6d}"
                f"{df['longitude'].iloc[i]:12.5f}"
                f"{df['latitude'].iloc[i]:12.5f}"
                f"{df['magnitude'].iloc[i]:12.1f}"
                f"{z_days.iloc[i]:12.5f}"      # ← day unit
                f"{-df['depth'].iloc[i]:8.2f}"
                f"{years[i]:12d}"
                f"{months[i]:3d}"
                f"{days[i]:3d}\n"
            )
            f.write(line)

    print("work.etas created")

def make_etas_open(flag, Mc):

    if flag == "old":
        start = datetime(1998, 1, 1)
        end   = datetime(2016, 3, 31)
    elif flag == "new":
        start = datetime(2016, 4, 1)
        end   = datetime(2023, 12, 31)

    x_days = (end - start).days

    with open("4_etas/etas.open", "w") as f:
        f.write("4         2\n")
        f.write(f"0.0       {x_days:.2f}     30.0\n")
        f.write(f"{Mc:.1f}       {Mc:.1f}\n")
        f.write("0.0100E+00 0.0100E+00 0.30000E-01 0.12000E+01 0.1100E+01\n") # initial values

def remake_etas_open(flag, Mc, x):

    if flag == "old":
        start = datetime(1998, 1, 1)
        end   = datetime(2016, 3, 31)
    elif flag == "new":
        start = datetime(2016, 4, 1)
        end   = datetime(2023, 12, 31)
    
    mu, K, c, alpha, p = x

    x_days = (end - start).days

    with open("4_etas/etas.open", "w") as f:
        f.write("4         2\n")
        f.write(f"0.0       {x_days:.2f}      0.0\n")
        f.write(f"{Mc:.1f}       {Mc:.1f}\n")
        f.write(f"{mu: .5E} {K: .5E} {c: .5E} {alpha: .5E} {p: .5E}\n")

def run_with_log(index, logfile="4_etas/etas.log"):
    log = open(logfile, "a")

    log.write(f"\n\n===== NEW RUN #{index}=====\n")
    log.flush()

    saved_stdout_fd = os.dup(1)

    try:
        os.dup2(log.fileno(), 1)
        x, g, f, aic = etas_module.run_etas()
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)
        log.close()

    return x, g, f, aic

def run_etas(mainshock_df, flag):

    if flag == "old":
        t0 = "1998-01-01"
        Mc = 0.8
    elif flag == "new":
        t0 = "2016-04-01"
        Mc = 0.5

    for i in range(len(mainshock_df)):
        print(f"\n================ RUN {i} ================")

        mainshock = mainshock_df.iloc[i]
        df = read_nearby_events(i, flag)
        if len(df) < 100:
            print("event number =", len(df))
            x = [np.nan, np.nan, np.nan, np.nan, np.nan]
            print("skip")
        else:
            print("event number =", len(df))

            make_work_etas(df, t0)
            make_etas_open(flag, Mc)

            x, g, f, aic = run_with_log(i)

            print("mu, K, c, alpha, p =", x)
            print("gradient", g)
            print("log-likelihood =", f)
            print("AIC =", aic)

            k = 0
            while (np.linalg.norm(g) > 1e-4) and (k<10):
                print("!!! not converge")
                remake_etas_open(flag, Mc, x)
                x, g, f, aic = run_with_log(i)
                print("event number =", len(df))
                print("mu, K, c, alpha, p =", x)
                print("gradient", g)
                print("log-likelihood =", f)
                print("AIC =", aic)
                k += 1
            if k == 10:
                x = [np.nan, np.nan, np.nan, np.nan, np.nan]
        

        with open("4_etas/etas_results.csv", "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([i, len(df), x[0], x[1], x[2], x[3], x[4]])
        
        mainshock_df.loc[i, "etas_mu"]    = x[0]
        mainshock_df.loc[i, "etas_K"]     = x[1]
        mainshock_df.loc[i, "etas_c"]     = x[2]
        mainshock_df.loc[i, "etas_alpha"] = x[3]
        mainshock_df.loc[i, "etas_p"]     = x[4]

    return mainshock_df


#if os.path.exists("4_etas/etas_results.csv"):
if 1 == 2:
    print('skip eastimaintg etas parameter')
else:
    with open("4_etas/etas_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "event_num", "mu", "K", "c", "alpha", "p"])
    mainshock_df_old = run_etas(mainshock_df_old, flag="old")
    mainshock_df_new = run_etas(mainshock_df_new, flag="new")
    print("running etas done")


def integrate_lambda(mainshock_row, df_nearby, Mc, T=20):

    mu    = mainshock_row["etas_mu"]
    K     = mainshock_row["etas_K"]
    c     = mainshock_row["etas_c"]
    alpha = mainshock_row["etas_alpha"]
    p     = mainshock_row["etas_p"]

    if np.isnan(mu):
        return np.nan

    t_main = mainshock_row["datetime"]

    integral = mu * T

    df = df_nearby[df_nearby["datetime"] < t_main].copy()

    times = (df["datetime"] - t_main).dt.total_seconds() / (24*3600)
    mags  = df["magnitude"].values

    for ti, Mi in zip(times, mags):
        A = K * np.exp(alpha * (Mi - Mc))

        if p != 1.0:
            integral += A *  (1 / (1-p)) * (
                (- ti + c)**(1-p) - (max(-T, ti) - ti + c)**(1-p)
            )

    return integral

def etas_foreshock_TF(mainshock_df, flag=""):
    if flag == "old":
        Mc = 0.8
    elif flag == "new":
        Mc = 0.5
    
    mainshock_df["tau_20days"] = np.nan
    p_value_list = []
    foreshock_list = []

    for i in tqdm(range(len(mainshock_df)), desc="Integrating λ (20 days)"):
        df_nearby = read_nearby_events(i, flag)

        tau_20days = integrate_lambda(
            mainshock_df.iloc[i],
            df_nearby,
            Mc,
            T=20
        )

        if np.isnan(tau_20days):
            p_value_list.append(np.nan)
            foreshock_list.append(np.nan)
        else:
            Nobs = mainshock_df.loc[i, "TF_stw count"]
            p_value = 1 - poisson.cdf(Nobs, tau_20days)
            p_value_list.append(p_value)
            foreshock_list.append(p_value<0.01)

        mainshock_df.loc[i, "tau_20days"] = tau_20days
    
    mainshock_df["etas_rate"] = p_value_list
    mainshock_df["TF_etas"] = foreshock_list
    return mainshock_df

mainshock_df_old = etas_foreshock_TF(mainshock_df_old, flag="old")
mainshock_df_new = etas_foreshock_TF(mainshock_df_new, flag="new")

mainshock_df_old.to_csv(f"mainshock_df_old.csv", index=False)
mainshock_df_new.to_csv(f"mainshock_df_new.csv", index=False)

def occurrence_rate():

    df_old = pd.read_csv("mainshock_df_old.csv")
    df_new = pd.read_csv("mainshock_df_new.csv")

    foreshock_list_old = df_old["TF_etas"]
    foreshock_list_new = df_new["TF_etas"]

    nan_old = foreshock_list_old.isna().sum()
    valid_old = foreshock_list_old.dropna()

    print(f"old :")
    print(f"NaN = {nan_old}")
    print(f"{np.sum(valid_old)}/{len(valid_old)} = "
          f"{100*np.sum(valid_old)/len(valid_old):.3f}%\n")

    nan_new = foreshock_list_new.isna().sum()
    valid_new = foreshock_list_new.dropna()

    print(f"new :")
    print(f"NaN = {nan_new}")
    print(f"{np.sum(valid_new)}/{len(valid_new)} = "
          f"{100*np.sum(valid_new)/len(valid_new):.3f}%\n")

    total_true = np.sum(valid_old) + np.sum(valid_new)
    total_valid = len(valid_old) + len(valid_new)

    print(f"old+new :")
    print(f"NaN = {nan_old + nan_new}")
    print(f"{total_true}/{total_valid} = "
          f"{100*total_true/total_valid:.3f}%")

occurrence_rate()