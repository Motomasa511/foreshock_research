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

df_old = pd.read_csv("df_lfe.csv", index_col=0, parse_dates=["datetime"])
make_work_etas(df_old, t0="1998-01-01")
