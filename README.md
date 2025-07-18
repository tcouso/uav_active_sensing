# uav_active_sensing

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

nstruction with a SSL reward model

This project was set in the context of using reinforcement learning for controlling unmanned aerial vehicles (UAVs). Specifically, we need to learn a policy that allows UAVs to scan a farmland territory in order to locate certain undesirable weeds. The problem encompasses:
- A set of N UAVs with downward pointing cameras that gives them a field of view of the surface.
- UAVs can move in three dimensions. Moving up increases the field of view span, but reduces resolution.
- While moving, UAVs collect views of the terrain, collectively constructing a sample of the territory.

<img width="1440" height="810" alt="image" src="https://github.com/user-attachments/assets/80a0e511-ce25-4363-8723-96a2f23dad50" />

We aim to train a sampling policy that allows UAVs to explore the image regions that contain the greater amount of novelty, since visiting those sections could be useful for exploration dependent tasks, such as detecting strange weeds or other relevant elements.
- We assume that the amount of novelty in an image section can be measured with **reconstruction accuracy**. That is, if a section of an image can be used to accurately reconstruct the full image, then that section is dense in information.
- Full image reconstruction from partial images can be done with pre-trained visual models.

In this implementation we use pre-trained **Masked Auto-Encoders (MAEs)** [https://arxiv.org/abs/2111.06377] as image reconstruction devices.
- UAVs take samples of a territory during their trajectories. The samples then are fed to MAE to reconstruct the entire territory.
- The global reward is proportional to the inverse of the MSE between the generated and the target map, so agents learn to maximize reconstruction accuracy.

<img width="694" height="374" alt="image" src="https://github.com/user-attachments/assets/60a91b8a-65c5-4674-8285-40fcbfd971de" />


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         uav_active_sensing and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── uav_active_sensing   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes uav_active_sensing a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

