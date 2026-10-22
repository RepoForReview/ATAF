import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

def paired_bh(data, reference):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data 必须是 pandas DataFrame。")
    if not data.columns.is_unique or reference not in data.columns:
        raise ValueError("列名必须唯一，且包含 reference。")

    values = data.to_numpy(dtype=float)
    if len(data) < 2 or not np.isfinite(values).all():
        raise ValueError("至少需要两行配对数据，且不能包含缺失或非数值数据。")

    baseline = data[reference].to_numpy(dtype=float)
    rows = []
    for method in data.columns:
        if method == reference:
            continue
        current = data[method].to_numpy(dtype=float)
        difference = current - baseline
        sd = difference.std(ddof=1)
        if np.all(difference == 0):
            t, p, dz = 0.0, 1.0, 0.0
        elif sd == 0:
            raise ValueError(f"{method} 的配对差值恒定且非零，无法进行常规 t 检验。")
        else:
            test = ttest_rel(current, baseline)
            t, p = float(test.statistic), float(test.pvalue)
            if not np.isfinite(t) or not np.isfinite(p) or p <= 0:
                raise ValueError(f"{method} 的检验出现数值异常或 p 值下溢，请检查原始数据。")
            dz = difference.mean() / sd
        rows.append({
            "method": method, "reference": reference, "n": len(baseline),
            "t": t, "p_raw": p, "Cohen_dz": dz,
        })

    if not rows:
        raise ValueError("至少需要一个比较算法。")
    result = pd.DataFrame(rows)
    p = result["p_raw"].to_numpy()
    order = np.argsort(p)
    adjusted = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    p_adjusted = np.empty(len(p))
    p_adjusted[order] = np.minimum(adjusted, 1.0)
    result["p_adjusted"] = p_adjusted
    return result


if __name__ == "__main__":
    import pandas as pd

    data = pd.DataFrame({
        "scratch": [8.053,10,9.172,9.568,7.916,9.176,7.613,7.574,8.707,7.861],
        "hyser": [6.125,9.368,6.746,9.293,4.783,6.381,6.738,5.549,7.391,6.485],
        "ninapro": [4.277,8.522,7.066,8.39,8.693,8.272,5.363,7.067,6.071,6.924],
        "senic": [4.768, 7.158,6.786,9.242,8.794,7.48,7.086,7.22,5.358,6.81]},
)

    result = paired_bh(data, reference="scratch")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.10g}"))
