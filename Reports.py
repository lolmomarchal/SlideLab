import os

"""
Generates summary and error reports for the preprocessing pipeline
"""
class Reports:
    def __init__(self, summary, error, path):
        self.summary = summary
        self.error = error
        self.path = path
        self.summary_path = os.path.join(self.path, "SummaryReport.txt")
        self.error_path = os.path.join(self.path, "ErrorReport.txt")
        self.summary_report()
        self.error_report()

    def summary_report(self):
        if os.path.isfile(self.summary_path):
            option = "w"
        else:
            option = "x"
        with open(self.summary_path, option) as file:
            file.write("Summary Report\n")
            for item in self.summary:
                file.write(
                    f"Patient ID : {item[0]} Path: {item[1]}\nTotal Tiles: {item[2]} Tiles After Blurry Filter {item[3]}\n-------------------------------------\n")

    def error_report(self):
        if os.path.isfile(self.error_path):
            option = "w"
        else:
            option = "x"
        with open(self.error_path, option) as file:
            file.write("Error Report")
            current_patient = None
            for item in self.error:
                if item[0] != current_patient:
                    current_patient = item[0]
                    file.write(f"\n-------------------------------------\n Patient ID: {item[0]}\n")
                file.write(f"Error: {item[2]}\nLocation: {item[4]}\n Path: {item[1]}\n")

