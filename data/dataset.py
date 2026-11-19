"""CAMELS-US Daymet reader and normalized sliding windows."""
from bisect import bisect_right
from itertools import accumulate

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def unique_file(folder, pattern):
    files = list(folder.glob("*/" + pattern))
    if len(files) != 1:
        raise ValueError(f"Expected one {pattern} under {folder}, found {len(files)}")
    return files[0]


def daily_data(root, basin, dates):
    forcing = unique_file(root / "basin_mean_forcing/daymet", basin + "_*.txt")
    streamflow = unique_file(root / "usgs_streamflow", basin + "_*.txt")
    with forcing.open() as header:
        next(header)
        next(header)
        area_km2 = float(next(header))
    met = pd.read_csv(forcing, sep=r"\s+", skiprows=3)
    q = pd.read_csv(streamflow, sep=r"\s+", header=None,
                    names=["gauge", "Year", "Mnth", "Day", "flow", "flag"])
    for table in (met, q):
        table.index = pd.to_datetime(dict(year=table.Year, month=table.Mnth, day=table.Day))
    fields = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"]
    discharge = 28316846.592 * q.flow * 86400 / (area_km2 * 1e6)
    series = met[fields].join(discharge.rename("runoff"), how="inner").dropna()
    subset = series.loc[dates[0]:dates[1]]
    subset = subset.loc[subset.runoff >= 0]
    return (subset[fields].to_numpy(dtype=np.float32),
            subset[["runoff"]].to_numpy(dtype=np.float32))


class CamelsDataset(Dataset):
    def __init__(self, root, ids, dates, train=None):
        self.basins_list = ids
        series = [daily_data(root, basin, dates) for basin in ids]
        self.length_ls = [len(y) - 21 for x, y in series]
        if min(self.length_ls) < 1:
            raise ValueError("Each gauge needs 22 valid days within the selected period.")
        self.ends = list(accumulate(self.length_ls))
        if train is None:
            all_x = np.concatenate([x for x, y in series])
            all_y = np.concatenate([y for x, y in series])
            self.stats = (np.nanmean(all_x, axis=0), np.nanstd(all_x, axis=0),
                          np.nanmean(all_y, axis=0), np.nanstd(all_y, axis=0))
            self.basin_std = [float(y.std()) for x, y in series]
        else:
            if train.basins_list != ids:
                raise ValueError("Training statistics must use the same gauge order.")
            self.stats, self.basin_std = train.stats, train.basin_std
        mx, sx, my, sy = self.stats
        self.series = [((x - mx) / sx, (y - my) / sy) for x, y in series]

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, idx):
        gauge = bisect_right(self.ends, idx)
        start = idx - (self.ends[gauge - 1] if gauge else 0)
        x, y = self.series[gauge]
        return (x[start:start+22], y[start:start+15], y[start+15:start+22], self.basin_std[gauge])

    def runoff_mm(self, values):
        return values * self.stats[3] + self.stats[2]

