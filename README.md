# Machine Learning-Assisted Spin-Glass Optimization

Group project for **Advanced Machine Learning for Physics**, Sapienza University of Rome.

**Authors:** Flavio Massimiliano Ricci, Assiem Vigiani and Friedrich Rauno Zander.

This project compares **Simulated Annealing (SA)**, **Population Annealing (PA)** and **Global Annealing (GA)** for finding low-energy configurations of disordered Ising spin systems. GA uses a MADE autoregressive neural model to propose global configurations, combined with local Monte Carlo updates and a Metropolis–Hastings acceptance rule.

## Scientific scope

The main benchmark is the three-dimensional **Edwards–Anderson (EA) spin glass**, with nearest-neighbour interactions and Gaussian random couplings. The notebook studies a cubic lattice with L = 10 and N = 1000 spins.

The project contains three parts:

1. **Annealing comparison:** selected comparisons inspired by the reference study, tracking energy gaps relative to reference energies supplied in Gurobi logs.
2. **Fixed-temperature sampling:** energy and overlap distributions at T = 0.77, including GA variants with local moves, learned global moves, and different retraining intervals.
3. **Preliminary SK extension:** a comparison on a fully connected **Sherrington–Kirkpatrick** instance with N = 512 spins.

This is an educational reproduction and exploration of published methods. The algorithms and their original research claims belong to the cited literature.

## Methods

| Method | Main components |
| --- | --- |
| Simulated Annealing | Cooling with local Metropolis updates |
| Population Annealing | Cooling, population resampling, and local Metropolis updates |
| Global Annealing | MADE training, global proposals, Metropolis–Hastings correction, local updates, and retraining |

The EA implementation uses checkerboard updates. The SK extension uses sequential single-spin updates. The implementations use Python and PyTorch, with population operations that can run on a GPU.

## Results and limitations

On the illustrated EA run, PA reaches the reference low-energy region earlier than GA, while both find lower final energies than SA. The notebook also includes fixed-temperature energy and overlap comparisons and the preliminary SK analysis.

These are observations from the studied runs, not a general ranking of the methods. The notebook uses different schedule lengths for some comparisons; the parameter cells document those choices. Runtime comparisons depend on hardware, settings and the treatment of initialization and training costs.

**Experiment settings:** The notebook and slides use different settings for the fixed-temperature GA comparison. The notebook's GA sampler-variant comparison uses one local sweep per global move, while the slides describe 15. The parameter cells associated with each notebook figure specify that experiment's configuration.

For the SK extension, no exact ground-state reference is available in this project. The plotted minimum is the lowest energy found during a run, not a certified ground state. Agreement with a sampling reference distribution also does not by itself establish equilibration.

## Repository contents

| File | Description |
| --- | --- |
| `annealing.ipynb` | Main notebook containing the algorithm implementations, experiments, plots and discussion. |
| `plot.py` | Plotting functions imported by the notebook. |
| `slides/presentation.pptx` | Presentation of the scientific background, methods and results. |
| `references.bib` | Bibliographic entry for the reference paper. |

The notebook and `plot.py` are located in the repository root. Benchmark data are retrieved separately from the upstream repository linked below.

## Running the notebook

The recorded runs used **Python 3.12.13**, **PyTorch 2.11.0+cu128** and an **NVIDIA Tesla T4 GPU**. Versions of the other dependencies were not recorded.

The code uses **PyTorch**, **NumPy**, **pandas** and **Matplotlib**. A local interactive run also needs a Jupyter notebook environment. A GPU is recommended for the recorded population sizes; CPU execution may take substantially longer.

1. Place the notebook and `plot.py` in the same directory.
2. Install the dependencies in your chosen Python environment, selecting a PyTorch build appropriate for your hardware.
3. Obtain the benchmark data from the upstream repository linked below. The notebook expects the upstream `Data` directory and its existing subdirectory layout.
4. Set the notebook's data paths to your local copy, or follow its Colab setup. The Colab plotting setup expects `plot.py` in `MyDrive/progetto_ML` or the current working directory.
5. Review the parameter cells, then run the notebook in order. Optimization, sampling, and SK experiments can be computationally expensive.

**Google Colab setup:** The notebook was developed for Google Colab. It looks for `plot.py` in the current working directory or `MyDrive/progetto_ML`, and uses `/content/MLMC_optimization` for the upstream benchmark repository. Keep this layout when using Colab; adjust the corresponding paths for local execution. The benchmark data are obtained separately from the upstream repository and are not bundled here.

## Reference paper

Luca Maria Del Bono, Federico Ricci-Tersenghi, and Francesco Zamponi. **Demonstrating real advantage of machine learning–enhanced Monte Carlo for combinatorial optimization.** *Proceedings of the National Academy of Sciences* **123**, e2534768123 (2026).

- [Published article](https://doi.org/10.1073/pnas.2534768123)
- [arXiv preprint, including version history](https://arxiv.org/abs/2510.19544)
- [Reference implementation and benchmark data](https://github.com/Laplaxe/MLMC_optimization)

Benchmark couplings and reference data come from the repository accompanying the paper. Upstream materials retain their original attribution and applicable terms.

See `references.bib` for the paper's bibliographic entry.

## Authorship and contributions

Flavio Massimiliano Ricci, Assiem Vigiani and Friedrich Rauno Zander completed this work jointly for the Advanced Machine Learning for Physics course at Sapienza University of Rome.
