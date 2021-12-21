from enum import unique
import altair as alt
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import normalized_mutual_info_score
from pathlib import Path


class Plotter():
    def __init__(self, results_folder=None):
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

    def scatter(self, data, labels):
        """
        Plots the data points.
        """
        raise NotImplementedError("Not implemented for this plotter.")

    def assignments(self, data, labels):
        """
        Plots the assignments of the data points to the clusters.
        """
        raise NotImplementedError("Not implemented for this plotter.")

    def comparison(self, data, true, predicted):
        """
        Compares predicted to true label.
        """
        raise NotImplementedError("Not implemented for this plotter.")

    def show(self):
        """
        Shows the plot.
        """
        raise NotImplementedError("Not implemented for this plotter.")

    def save(self):
        """
        Saves the plot.
        """
        raise NotImplementedError("Not implemented for this plotter.")


class AltairPlotter(Plotter):
    def __init__(self, results_folder=None):
        super().__init__(results_folder=results_folder)
        self.last_chart = None

    # Decorator for cleanup
    def _altair_plotting_function(func):
        def wrapper(self, *args, **kwargs):
            chart = func(self, *args, **kwargs)
            self.last_chart = chart
            return chart
        return wrapper

    @_altair_plotting_function
    def scatter(self, data):
        """
        Plots the data points.
        """
        data = pd.DataFrame(data, columns=["x", "y"])

        chart = alt.Chart(data).mark_point().encode(
            x="x",
            y="y",
        )
        return chart

    @_altair_plotting_function
    def assignments(self, data, labels):
        """
        Plots the assignments of the data points to the clusters.
        """
        data = pd.DataFrame(data, columns=["x", "y"])
        data["predicted"] = labels

        chart = alt.Chart(data).mark_point().encode(
            x="x",
            y="y",
            color=alt.Color("predicted:N", scale=alt.Scale(scheme="dark2"))
        )
        return chart

    @_altair_plotting_function
    def comparison(self, data, true, predicted):
        """
        Compares predicted to true label.
        """
        chart1 = self.assignments(data, true).properties(title="True labels")
        chart2 = self.assignments(data, predicted).properties(
            title="Predicted labels")
        combined = chart1 | chart2
        combined = combined.properties(
            title=f"Clustering comparison, NMI = {normalized_mutual_info_score(true, predicted):.2f}")
        return combined

    def show(self):
        """
        Shows the plot.
        """
        if self.last_chart is not None:
            self.last_chart.show()
        else:
            raise ValueError("No chart to show.")

    def save(self, name):
        if self.last_chart is None:
            raise ValueError("No chart to show.")
        if self.results_folder is None:
            raise ValueError("No results folder to save to.")
        self.last_chart.save(str(self.results_folder / name))
