import os
import pandas as pd
import numpy as np

"""
Generates summary and error reports for the preprocessing pipeline

Outputs: A summary of the processed slides and a report of any error encountered. 

Summary reports include:
- Sample ID, Path to Slide,number of tiles found, number of tiles that pass the QC, and percentage of tissue tiles remaining 

Error reports include:
- Sample ID, Path to Slide, Error type/message, Step in line where the error occurred.

"""


class Reports:
    def __init__(self, summary, error, path):
        self.summary = summary
        self.error = error
        self.path = path
        self.summary_path = os.path.join(self.path, "SummaryReport.csv")
        self.error_path = os.path.join(self.path, "ErrorReport.csv")
        self.summary_report()
        self.error_report()

    def summary_report(self):
        if not self.summary[0]:
            return
        columns = ["sample_id", "path", "total_tiles", "tiles_passing_tissue_thresh", "non_blurry_tiles",
                   "time_masking_cpu", "time_obtaining_valid_coordinates_cpu",
                   "time_patching_tiles_cpu", "time_masking_user", "time_obtaining_valid_coordinates_user",
                   "time_patching_tiles_user", "status"]
        summary = pd.DataFrame(self.summary, columns=columns)
        # print(summary)
        summary["%_tissue"] = (summary["tiles_passing_tissue_thresh"] / summary["total_tiles"]) * 100
        summary["%_tiles_passing_QC"] = summary.apply(
            lambda row: (row["non_blurry_tiles"] / row["tiles_passing_tissue_thresh"] * 100)
            if row["non_blurry_tiles"] is not None and row["tiles_passing_tissue_thresh"] not in (None, 0)
            else None,
            axis=1,
        )
        summary["total_time_cpu"] = summary["time_patching_tiles_cpu"] + summary["time_masking_cpu"] + summary[
            "time_obtaining_valid_coordinates_cpu"]
        summary["total_time_user"] = summary["time_patching_tiles_user"] + summary["time_masking_user"] + summary[
            "time_obtaining_valid_coordinates_cpu"]
        # empty values -> error occurred so
        summary = summary.fillna("N/A")

        if os.path.exists(self.summary_path):
            existing_summary = pd.read_csv(self.summary_path)
            existing_summary["status"].astype(str)
            processed_entries = existing_summary.loc[
                existing_summary["status"] == "Processed", "sample_id"
            ]
            summary = summary[~summary["sample_id"].isin(processed_entries)]
            combined_summary = pd.concat([existing_summary, summary], ignore_index = True)
            combined_summary = combined_summary.fillna("N/A")
            combined_summary.to_csv(self.summary_path, index=False)
        else:
            summary.to_csv(self.summary_path, index=False)
    def summary_report_update_encoding(self,encoding_data):
        existing_summary = pd.read_csv(self.summary_path)
        encoding_summary = pd.DataFrame(encoding_data, columns = ["sample_id", "encoding_cpu_time", "encoding_user_time"])
        if "encoding_cpu_time" not in list(existing_summary.columns):
            merged = existing_summary.merge(encoding_summary, on="sample_id", how="left")
        else:
            merged = existing_summary.merge(encoding_summary, on="sample_id", how="left", suffixes=("", "_new"))
            for col in ["encoding_cpu_time", "encoding_user_time"]:
                merged[col] = merged[col].where(merged[f"{col}_new"] == -1, merged[f"{col}_new"])
            merged = merged.drop(columns=[f"{col}_new" for col in ["encoding_cpu_time", "encoding_user_time"]])
        merged.to_csv(self.summary_path, index=False)



    def error_report(self):
        if self.error[0]:
            columns = ["Sample ID", "Path", "Error", "Step in Pipeline"]
            error = pd.DataFrame(self.error[0], columns=columns)
            if os.path.exists(self.error_path):
                existing_error = pd.read_csv(self.error_path)
                combined_error = pd.concat([existing_error, error]).drop_duplicates()
                combined_error.to_csv(self.error_path, index=False)
            else:
                error.to_csv(self.error_path, index=False)
