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
        columns = ["Sample ID", "Path", "Total Tiles", "Non-blurry Tiles"]
        summary = pd.DataFrame(self.summary, columns=columns)
        summary["Percentage of Tiles Passing QC"] = (summary["Non-blurry Tiles"] / summary["Total Tiles"]) * 100
        summary.to_csv(self.summary_path, index=False)

    def error_report(self):
        columns = ["Sample ID", "Path", "Error", "Step in Pipeline"]
        error = pd.DataFrame(self.error, columns=columns)
        error.to_csv(self.error_path, index=False)