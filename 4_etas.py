# for 本震:
    # 本震周りの全期間の地震をdf_nearbyに
    # work.etasとetas.openを作る
    # パラメータフィット(etas.f)しmainshock_dfに追加
    # 20daysでのetasでの地震数を求め、Nobsがポアソン分布のxx%にあるxxを求める
    # xx <0.01 で T or F

# df_nearby_list を保存
# rate を求める


import pandas as pd
import numpy as np
from datetime import datetime
import os

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

    return nearby_event_dfs

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

def make_etas_open(mainshock_df, flag, Mc):

    if flag == "old":
        start = datetime(1998, 1, 1)
        end   = datetime(2016, 3, 31)
    elif flag == "new":
        start = datetime(2016, 4, 1)
        end   = datetime(2023, 12, 31)
    else:
        raise ValueError("flag must be 'old' or 'new'")

    x_days = (end - start).days

    for _, row in mainshock_df.iterrows():
        number = row["number"]

        #with open("4_etas/etas.open", "w") as f:
        with open("etas.open", "w") as f:
            f.write("9         2\n")
            f.write(f"0.0       {x_days:.2f}     30.0\n")
            f.write(f"{Mc:.1f}       {Mc:.1f}\n")
            f.write("0.50000E+00 0.63348E+02 0.38209E-01 0.26423E+01 0.10169E+01\n") # initial values


df_old = pd.read_csv("df_lfe.csv", index_col=0, parse_dates=["datetime"])
make_work_etas(df_old, t0="1998-01-01")
make_etas_open(mainshock_df_old[0:1], flag="old", Mc=0.6)
