# Computational-Neurodynamics
Coursework for Computational Neurodynamics Module @ Imperial College

## About
This project simulates a Small World Modular Network of **Izhikevich Neurons**. It follows the Network creation and rewiring procedures covered in Lectures 8 (Modular Networks) and 9 (Dynamical Complexity).

Information and specifications about the class parameters and functions can be found in the _modularnet.py_ file.
    
## Getting Started
To get started, create an object from the **Small_World_Modular_Net** class and set the required parameters, then simulate the network for a desired amount of time.

By default, the class generates a Network of 1000 Izhikevich Neurons organized into 8 Communities, each with 100 _Excitatory Neurons_ and 1000 random intra-edges, and a core of 200 _Inhibitory Neurons_ connected to every Network Neuron.

Here is an example use of the class to create a default Network with no rewiring, and simulate it for 1000ms:

    from iznetwork import IzNetwork
    from modularnet import Small_World_Modular_Net

    net = Small_World_Modular_Net(p=0, n=1000, C=8, m=1000) 
    net.plot_weights("Matrix Connectivity")
    net.simulate_net(T=1000) # simulate network activity for 1 second
    net.plot_raster(T=1000) # generate raster plot
    net.plot_rolling_mean_per_module(T=1000) # firing rate plot

## Dependencies
The project requires the following packages in order to work:
* **python**: The used programming language (any version >= 3.9).
* **iznetwork**: The python file for the IzNetwork class that simulates the Izhikevich Neurons (provided alongside coursework specifications). It needs to be in the same directory as the _modularnet.py_ file for the network simulations to run.
* **numpy**: This package is used throughout the project to perform all of the vector/matrix operations.
* **matplotlib**: This package is used to generate the raster plots.
* **seaborn**: This package is used to generate the heatmap of the network matrix connectivity.