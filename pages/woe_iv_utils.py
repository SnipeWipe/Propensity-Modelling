import pandas as pd
import numpy as np


def calculate_iv(df, target):

    iv_list = []

    features = [col for col in df.columns if col != target]

    for feature in features:

        temp = pd.DataFrame({
            feature: df[feature],
            target: df[target]
        })

        # Convert numeric variables into bins
        if pd.api.types.is_numeric_dtype(temp[feature]):
            try:
                temp[feature] = pd.qcut(
                    temp[feature],
                    q=10,
                    duplicates="drop"
                )
            except:
                pass

        grouped = temp.groupby(feature)[target].agg(
            total="count",
            bad="sum"
        ).reset_index()

        grouped["good"] = (
            grouped["total"] - grouped["bad"]
        )

        grouped["good_dist"] = (
            grouped["good"] /
            grouped["good"].sum()
        )

        grouped["bad_dist"] = (
            grouped["bad"] /
            grouped["bad"].sum()
        )

        grouped["WOE"] = np.log(
            (grouped["good_dist"] + 1e-6) /
            (grouped["bad_dist"] + 1e-6)
        )

        grouped["IV"] = (
            grouped["good_dist"] -
            grouped["bad_dist"]
        ) * grouped["WOE"]

        iv = grouped["IV"].sum()

        iv_list.append({
            "variable": feature,
            "info_value": round(iv, 4)
        })

    return pd.DataFrame(iv_list)
