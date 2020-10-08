# Taxi-time prediction - Group 2

Welcome to the repository hosting the code of group 2 for the Eleven Data Challenge on Taxi-time prediction. 

## Repository structure

We splited our our into notebooks and python files. The python files contain all the preprocessing and the modeling while the notebooks were used for experimentation and contain exploratory data analysis and interpretability related code and visuals. You will find below the structure of our repo to walk you through the code and a general case description containing additional info about the case itself. 

### ``main.py``

The gateway to our code is the main.py file. It takes one command line argument ``--dataset_loaded`` indicating if the preprocessed train and test datasets are available. If this flag is up, we import the packages, skip the preprocessing steps and go directly to step number 6 below. Otherwise we do the following in main.py:

1. Importing the necessary packages
2. Launching the Preprocessing pipelines, obtaining clean datasets and combining the datasets
3. Splitting into training and validation datasets
4. Models definition

### Preprocessing folder

We perform all our preprocessing using classes defined in the preprocessing folder:
- **preprocessing.py:** This file contains definitions of the preprocessing piplines for each dataset.
- **data_imputer.py:** This file contains the imputers for each pipeline whose job is to clean already existing features and apply modifications to them.
- **data_augmenter.py:** This file contains the augmenters whose job is to build new features into our datasets.
- **data_encoder.py:** This file is used to encode the features that require encoding.

### Modeling folder
Each file in this folder contains a class defining one of the models we used.
Hyperparameter tuning was done either one Dataiku DSS or a seperate notebook.

### Utils folder
Utility methods to be used by the preprocessing pipelines and metrics for our models.

## Case description

- **Preliminary info:** The taxi-time is the time an airplane spends “driving” on the ground:
    - Taxi-in is the time window between the moment the airplane’s wheels touch the ground i.e. the Actual Landing Time (ALDT) and the moment it arrives at its assigned dock i.e. Actual In-Block Time (AIBT)
    - Taxi-out is the time window between the moment the airplane starts moving from its dock i.e. Actual Off-Block Time (AOBT) to the moment its wheels leave the ground i.e. Actual Take-Off Time (ATOT)

- **Objective:** The goal of this case is to rovide an accurate Take-Off Time (ATOT) prediction based on an actual off-block time (AOBT) and an algorithm-based taxi-out time prediction considering factors such as airport configuration, AC type, weather etc

- **Importance:** The reason why this prediction is so important for airports is that all ground operations (the operations that happen when the airplane reaches its dock e.g. fueling, catering, cleaning...) depend on the precision whereby the ground handling teams are dispatched.For instance, let’s say that the predicted taxi-time for a plane set to dock at gate A1 is 25min while the actual taxi-time is 5min. The prediction error implies that ground-handling teams are idle at dock A1 for 20min and that they will not be available to handle the departure of other planes  which  could  have  left  earlier.  This  in  turn  generates  delays,  and  overall  economic inefficiencies for both the airport and the ground-handlers

- **Challenge:** Use open source packages and libraries to develop a predictive algorithm for airplanes’ taxi-time.
   
- **Data (type, volume):** Data provided by eleven includes:
    - Real airport operational data
    - Official FAA aircraft characteristics
    - Weather data from the considered airport

