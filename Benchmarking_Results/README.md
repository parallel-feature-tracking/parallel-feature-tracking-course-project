
# Setting Up a Python Virtual Environment

This guide will show you how to create a virtual environment for the Python project. A virtual environment is a tool that helps to keep dependencies required by different projects separate by creating isolated python virtual environments for them.

## Prerequisites

Ensure you have Python installed on your system. The `virtualenv` package works with Python 2.7+ and Python 3.3+.

## Installation

First, install the `virtualenv` package globally using pip:

```bash
pip install virtualenv
```



# Create a new python virtual env

```bash
python3 -m venv pfet_env
```

```bash
source pfet_env/bin.activate
```

Before running these scripts, ensure the following requirements are met:
- Python 3.x installed
- Necessary libraries installed (e.g., `numpy`, `matplotlib`,etc)

# Benchmarking Scripts for Parallel Feature Extraction and Tracking

The script for running the profilers and the experiments is given in the parent directory. Kindly use the profiler html and xml's generated there to run the scripts. Otherwise they are included in the given drive link [here](https://iitk-my.sharepoint.com/:f:/g/personal/spratham21_iitk_ac_in/EvVUBMmVfVhJornbDR-fhYQBCxN1mqa4D7ymEFaSkCVj8w?e=Noi73w)
This repository contains a suite of Python scripts used for the benchmarking of parallel feature extraction and tracking algorithms as described in the accompanying paper.

## Scripts Overview

Each script in this repository serves a specific purpose in the benchmarking process. Below is a description of what each script does:

### `parser_pfet.py`
Parses profiler htmls for the overview of all the experiment runs.

### `percent_commtime.py`
Analyzes the percentage of communication time with respect to dataset size and number of processes spawned.

### `plot_data_processes.py`
Generates a graph for wallclock time taken for a given dataset size in across different number of processes.

### `plot_mpi_proc_log.py`
Creates log-scaled plots related to different MPI processes and how much time they take to execute.

### `plot_proc_datasize.py`
Visualises the time taken by a given process count run on different sizes of datasets.

### `scrape_xml_folder.py`
Scrapes XML files from a specified directory, processing them for use in subsequent benchmarking steps. It returns a csv output.


