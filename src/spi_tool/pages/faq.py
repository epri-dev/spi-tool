import panel as pn
import scipy
import numpy as np
import matplotlib as mpl


class FAQ(pn.viewable.Viewer):
    def _update_distribution_plots(self):
        fig = mpl.figure.Figure(figsize=(10, 5))
        axs = fig.subplots(1, 2, sharey=True, sharex=True)
        ax = axs[0]
        x = np.linspace(-5, 5, 100)
        ax.plot(x, scipy.stats.norm.pdf(x, 0, 1))
        ax.set_title("Normal Distribution")
        ax.set_xlabel("Variable range -∞ to +∞")

        ax = axs[1]
        x = np.linspace(0, 5, 100)
        ax.plot(x, scipy.stats.lognorm.pdf(x, 1, 0))
        ax.set_title("Lognormal Distribution")
        ax.set_xlabel("Variable range 0 to +∞")
        return fig

    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown(
                """
# Frequently Asked Questions

This section provides answers to common questions about SPI-Tool.

## What is SPI-Tool?

SPI-Tool is a web-based application that helps resource planners characterize future uncertainty in the production cost modeling inputs used in a Monte Carlo study in an Integrated Resource Planning (IRP) process.

## What are Auto-Regressive Models and why does SPI-Tool use them?

Auto-regressive models are a common approach to creating Monte Carlo time-series samples for a stochastic analysis for IRP. Auto-regressive models predict a future value as a linear function of that variable's past values. Auto-regressive models are capable of modeling behaviors found in many drivers of uncertainty:

- **Autocorrelation**: when correlation exists between consecutive time periods in a time series. For example, natural gas prices today tend to be like yesterday's.
- **Mean-reversion**: when values in a time series tend to move back towards an average value over time. If a process experiences a shock (e.g., geopolitical forces causing a spike in natural gas prices), mean-reversion occurs if the process tends to return to normal over time.

AR(1) models are a special form of auto-regressive models that only use the random variable's most recent value to generate the variable's future value.

Auto-regressive models are well-suited for modeling stochastic inputs with intra-annual uncertainty that exhibit the above behaviors: load, commodity prices (i.e., coal, natural gas, fuel oil), and sometimes hydro-electric generation.

Examples of IRPs that used auto-regressive models to generate their Monte Carlo samples in their stochastic analyses include:

- PacifiCorp. “2023 Integrated Resource Plan,” March 31, 2023.
- AES Indiana. “2022 Integrated Resource Plan,” December 1, 2022.
- CenterPoint Energy. “2022/2023 Integrated Resource Plan,” 2023.
- Idaho Power. “2023 Integrated Resource Plan,” September, 2023.
- Tennessee Valley Authority. “2019 Integrated Resource Plan,” 2019.

## When should I use a normal distribution versus a lognormal distribution?

First, it is helpful to take a look at the differences between normal and lognormal distributions:

""",
                max_width=800,
            ),
            pn.pane.Matplotlib(self._update_distribution_plots(), height=400),
            pn.pane.Markdown(
                """

Use a normal distribution when:

- Data is symmetric and centered around a mean
- Negative values are possible
- The mean, median, and mode are about the same

Use a lognormal distribution when:

- Data is positively skewed (right-tailed) with no negative values
- Small values are common, but large values can occur occasionally
- If taking the logarithm of your data into a normal distribution, a lognormal distribution might be a good fit!

## What does the random seed do?

The random seed is a number that initializes the random number generator. It is used to ensure that the same sequence of random numbers is generated each time the program is run with the same seed. This is useful for reproducibility, as it allows you to generate the same random samples each time you run the program.
The random seed is not required, but it is recommended to set it to a specific value if you want to reproduce the same results in future runs. If you wish to generate different results, use the button "New Random Seed" to generate a new random seed.

## I see warnings on the terminal

The warnings are harmless and are due to the way the application is designed. For example, the following warning is harmless:

```
WARNING:root:Dropping a patch because it contains a previously known reference (id='p1624'). Most of the time this is harmless and usually a result of updating a model on one side of a communications channel while it was being removed on the other end.
```

Please feel free to report any errors or issues you encounter via email.

            """,
                max_width=800,
                sizing_mode=None,
            ),
            sizing_mode=None,
        )
