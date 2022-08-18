## Estimating True Number of COVID-19 Incidence 

This repository contains Python code to estimate the true number of covid-19 infections. The methodology follows from what is described in this [paper](https://www.acpjournals.org/doi/10.7326/M21-2721).

### How to run 
The main requirement is `python 3.8 3.9, or 3.10`. Use `pip` to install the required dependencies. Run the main script by `python3 main.py`. 

Alternatively, a Docker file is provided. In terminal, enter commands:
```
docker build --tag truecount . 
docker run truecount
```




