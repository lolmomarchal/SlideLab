import os
import pandas as pd

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
        columns = ["sample_id", "path", "total_tiles", "tiles_passing_tissue_thresh","non_blurry_tiles",
                   "total_processing_time_cpu", "time_opening_slide_cpu", "time_masking_cpu", "time_obtaining_valid_coordinates_cpu",
                   "time_patching_tiles_cpu",  "total_processing_time_user", "time_opening_slide_user", "time_masking_user", "time_obtaining_valid_coordinates_user",
                   "time_patching_tiles_user"]
        summary = pd.DataFrame(self.summary, columns=columns)
        summary["Percentage of Tissue Tiles Passing QC"] = (summary["non_blurry_tiles"] / summary["tiles_passing_tissue_thresh"]) * 100
        summary["Percentage of Tissue Tiles"] = (summary["tiles_passing_tissue_thresh"] / summary["total_tiles"]) * 100
        # empty values -> error occurred so
        summary = summary.fillna("N/A")
        summary.to_csv(self.summary_path, index=False)

    def error_report(self):
        columns = ["Sample ID", "Path", "Error", "Step in Pipeline"]
        error = pd.DataFrame(self.error, columns=columns)
        error.to_csv(self.error_path, index=False)
