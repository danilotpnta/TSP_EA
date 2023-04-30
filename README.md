# TSP_EA

Below the performance of the algorithm. In the green line is the heuristics to beat. It can be seen that the algorithm performs well during all the duration on the 5min time contraint. On the blue line the Mean Fitness its plotted. From the oscilating peaks it is shown that the EA does exploration throughout all the cycle while conserving the optimal minium (Best Fitness) at a constant pace. At the beginning of initTime (left graph) it is shown that the mutation mechanisms allow to explore all the search space. Thus, avoiding oscilating around a local minimum.


<p align="left">
  <img src="https://github.com/danilotpnta/TSP_EA/blob/master/img/100back.gif">
</p>

Furthermore to sustain proof for replicability this EA algorithm was run over 300 iterations (each iteration having a run time of 5min) where the Best and Mean Fitness values where stored and then plotted in a histogram. The following histogram (right side) shows the Best Fitness is far superior from its neraby peaks. Thus concluding the algorithm performs at its best regardless of different and randomize search spaces (different csv. files)

<p align="left">
  <img src="https://github.com/danilotpnta/TSP_EA/blob/master/img/Histogram.png">
</p>

This project was awarded a 20/20 grade. For further implementation of the different phases that made this project visit [r0916799.py](https://github.com/danilotpnta/TSP_EA/blob/master/r0916799.py). Following a sneak peak on the features of this EA. 

<p align="center">
  <img src="https://github.com/danilotpnta/TSP_EA/blob/master/img/EA_algorithm.png" width="300">
</p>
