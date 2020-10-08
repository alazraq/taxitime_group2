# Taxi-time prediction - Group 2

Welcome to the repository hosting the code of group 2 for the Eleven Data Challenge on Taxi-time prediction. You will find below the structure of our repo to walk you through the code and a general case description containing additional info about the case. 

## Repository structure

### ``main.py``

The gateway to our code is the main.py file.


## Preprocessing folder

## Modeling folder

## Utils folder


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


- **Code eliverables:** Code should include:
    - Your feature engineering code specifying how you modified your data and why (make sure to clearly comment your code to explain why you processed the data the way you chose to)
    - Your models’ parametrization, training code and testing code

