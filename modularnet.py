"""
--- Computational Neurodynamics Coursework ---

This project simulates a Small World Modular Network of Izhikevich Neurons. 
It follows the Network creation and rewiring procedures covered in Lectures 
8 (Modular Networks) and 9 (Dynamical Complexity).

Asia Belfiore, Ginevra Cepparulo, Vincent Lefeuve
Imperial College London, Department of Computing, Autumn Semester 2024
"""

from iznetwork import IzNetwork
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Small_World_Modular_Net(object):
    """
    This Class simulates a Small World Modular Network of Izhikevich Neurons from the
    IzNetwork class. 
    It creates a network of 80% excitatory neurons grouped into modules, and a common core 
    of 20% inhibitory neurons. The network has directed edges between neurons and 
    there are no self-connections. 
    
    The network edges are represented as a numpy array of size(n,n), with n=total number 
    of neurons in the network. Each entry (i,j) can be:
        - 1, if there is a directed connection from neuron i to neuron j.
        - 0, otherwise.
    
    The modules are represented as a numpy array of size(C,e), with C=number of modules, 
    e=number of excitatory neurons in each module. Each module has a chosen (m) number of 
    intra-connections, randomly distributed between its neurons.
    Each entry (c,i,j) can be:
        - 1, if there is a directed connection from neuron i to neuron j within module c.
        - 0, otherwise. 
    Each edge in each module is rewired with probability p to an excitatory neuron of a
    different community.

    Each neuron in the inhibitory core is connected to every neuron in the network. Each
    inhibitory neuron also receives 4 incoming edges from randomly selected excitatory
    neurons within the same module.

    All class members and setup methods are hidden (underscored) and should not be called.
    Only the functions to assert, simulate and plot the network are public and can be called.
    """

    def __init__(self, p, n=1000, C=8, m=1000):
        """
        Initialize the Small World Modular Network with n neurons grouped into C modules, 
        each with m edges, and rewiring probability p.
        Inputs:
            - p -- Rewiring Probability
            - n -- Number of total Neurons in the Network (Defaults to 1000). Automatically 
                   splits them into 80% excitatory and 20% inhibitory neurons.
            - C -- Number of Communities (Defaults to 8)
            - m -- Number of Edges per Module (Defaults to 1000)
        """
        self._num_neurons = n
        self._D = np.zeros((n, n)) # Delays
        self._W = np.zeros((n, n)) # Weight Connections

        # list of indices of each neuron type
        self._excitatory = [i for i in range(int(n * 0.8))] # indices [0,800) = excitatory
        inhibitory_offset = len(self._excitatory) # index of first inhibitory neuron
        self._inhibitory = [(i + inhibitory_offset) for i in range(int(n * 0.2))] # indices [800,1000) = inhibitory

        # connection matrix: each (i, j) entry represents the directed connection from neuron i to neuron j
        self._connections = np.zeros((n, n))
        neuro_per_module = len(self._excitatory)//C
        self._num_modules = C
        # C communities of n//C excitatory neurons: each entry (c, i, j) represents the edge from the 
        # i-th to the j-th excitatory neurons in c-th community 
        self._communities = np.zeros((self._num_modules, neuro_per_module, neuro_per_module))

        # set neuron connections
        self._set_excitatory(m, neuro_per_module=neuro_per_module)
        self._set_inhibitory(exc_to_inhib=4, neuro_per_module=neuro_per_module)
        # ensure no self-connections
        for i in range(n):
            if self._connections[i, i] != 0:
                # print(f"WARNING: Self connection for neuron {i}")
                self._connections[i, i] = 0
        
        # OPTIONAL: check that all connections are valid based on Coursework Specifications
        self.assert_connections(neuro_per_module=neuro_per_module)

        # rewire connections
        self._p = p # rewiring probability
        self._rewire(p, neuro_per_module=neuro_per_module)

        # scale connections and set connection delays
        self._W = self._scale_connections(self._connections, inhibitory_offset=inhibitory_offset)
        self._set_delays(inhibitory_offset=inhibitory_offset)   

        # create network
        self._net = IzNetwork(N=n, Dmax=50)
        self._net.setDelays(self._D.astype(int))
        self._net.setWeights(self._W)
        a,b,c,d = self._set_params(inhibitory_offset=inhibitory_offset)
        self._net.setParameters(a,b,c,d)

        # (Tasks b,c) Store Fire Behaviour of Network
        self._fired = np.zeros((1000, self._num_neurons))

        # set plot parameters
        self._set_up_plotting()


    def _set_excitatory(self, m, neuro_per_module):
        """
        For each module, set 1000 random directed connections between excitatory neurons within 
        the same community of weight 1.
        Inputs:
            - m -- Number of Edges per Module (Defaults to 1000)
            - neuro_per_module -- Number of (Excitatory) Neurons in each Module
        """
        for module in range(self._num_modules):
            set_edges = 0
            connections = []
            # NB: The same excitatory neuron pair is represented in two ways (indices start at 0 = shift by 1): 
            #     - in self._communities, each neuron has indices in [0, e), with e=number of neurons per module:
            #       (c, i, j) = (2, 3, 90), i.e. in the third community, the 4th neuron connects to the 91st neuron
            #     - in self._connections, each neuron has indices in [0, n):
            #       (i, j) = (203, 290), i.e. the 204th neuron connects to the 290th neuron in the network
            #     Simply go from the first to the second representation by adding the module offset to the indices.
            #     - module offset = module number * number of neurons per module (c*e) 
            offset = module * neuro_per_module 
            while set_edges < m:
                source_neuro = np.random.randint(0, neuro_per_module)
                target_neuro = np.random.randint(0, neuro_per_module)
                # ensure no duplicate or self connections
                if (source_neuro != target_neuro) and ((source_neuro, target_neuro) not in connections):
                    self._communities[module, source_neuro, target_neuro] = 1
                    self._connections[source_neuro+offset, target_neuro+offset] = 1
                    connections.append((source_neuro, target_neuro))
                    set_edges += 1


    def _set_inhibitory(self, exc_to_inhib, neuro_per_module):
        """
        Set random directed incoming connections (based on uniform distribution in [0,1]) from excitatory neurons 
        within the same community to each inhibitory neuron. 
        It also sets random directed outgoing connections (based on uniform distribution in [-1,0]) from each 
        inhibitory neuron to all neurons in the network (except itself).
        Inputs:
            - exc_to_inhib -- Number of incoming excitatory edges for each inhibitory neuron
            - neuro_per_module -- Number of (Excitatory) Neurons in each Module
        """
        for inhib in self._inhibitory:
            set_edges = 0
            connections = []
            source_module = np.random.randint(self._num_modules) # randomly select a module
            while set_edges < exc_to_inhib:
                start_neuro = source_module * neuro_per_module
                end_neuro = start_neuro + neuro_per_module
                excitatory_source = np.random.randint(start_neuro, end_neuro)
                if (excitatory_source, inhib) not in connections:
                    self._connections[excitatory_source, inhib] = np.random.uniform(0, 1)
                    connections.append((excitatory_source, inhib))
                    set_edges += 1
            # each inhibitrory neuron connects to every neuron
            self._connections[inhib, :] = np.random.uniform(-1, 0, self._num_neurons)
            self._connections[inhib, inhib] = 0 # except itself


    def _scale_connections(self, connections, inhibitory_offset):
        """ 
        Scale each connection weight based on the type of connection (Excitatory to Excitatory, 
        Excitatory to Inhibitory, Inhibitory to Excitatory). 
        Inhibitory to Inhibitory connections are not scaled (weight=1).
        Inputs:
            - connections -- connection matrix
            - inhibitory_offset -- Index of first inhibitory neuron in the network
        """
        W = np.copy(connections)
        # Apply scaling factor to the weights
        W[:inhibitory_offset, :inhibitory_offset] *= 17 # Excitatory to Excitatory
        W[:inhibitory_offset, inhibitory_offset:] *= 50 # Excitatory to Inhibitory
        W[inhibitory_offset:, :inhibitory_offset] *= 2  # Inhibitory to Excitatory
        return W


    def _set_delays(self, inhibitory_offset):
        """ 
        Set delays based on the type of connection (Excitatory to Excitatory, Excitatory to Inhibitory, 
        Inhibitory to Excitatory and Inhibitory to Inhibitory).
        Inputs:
            - inhibitory_offset -- Index of first inhibitory neuron in the network
        """
        # Excitatory to Excitatory
        self._D[:inhibitory_offset, :inhibitory_offset] = np.random.randint(1, 20, (inhibitory_offset, inhibitory_offset)) 
        self._D[:inhibitory_offset, inhibitory_offset:] = 1 # Excitatory to Inhibitory
        self._D[inhibitory_offset:, :inhibitory_offset] = 1 # Inhibitory to Excitatory
        self._D[inhibitory_offset:, inhibitory_offset:] = 1 # Inhibitory to Inhibitory


    def _rewire(self, p, neuro_per_module):
        """ 
        Rewire each excitatory (to excitatory) connection in each module of the network with probability p. 
        Each rewired edge is deleted and replaced by a new connection to a randomly selected excitatory neuron 
        outside the source module.
        Inputs:
            - p -- Rewiring Probability
            - neuro_per_module -- Number of (Excitatory) Neurons in each Module
        """
        # Get the intra connections for each module
        for module in range(self._num_modules):
            offset = module * neuro_per_module
            intra_connection = np.argwhere(self._communities[module] > 0)
            # Evaluate each existing intra community connection
            for connection in intra_connection:
                if np.random.rand() < p:
                    # Remove the intra connection
                    source_neuro = connection[0]+offset
                    self._connections[connection[0]+offset, connection[1]+offset] = 0
                    self._communities[module, connection[0], connection[1]] = 0
                    # Add a new connection (anything outside the module)
                    target_neuro = np.random.randint(0, len(self._excitatory))
                    target_module = target_neuro // neuro_per_module
                    # if the target neuron is in the same module, keep generating a new neuron
                    while target_module == module:
                        target_neuro = np.random.randint(0, len(self._excitatory))
                        target_module = target_neuro // neuro_per_module
                    # print(f"remove connection from {connection[0]+offset} to {connection[1]+offset}")
                    # print(f"rewire connection from {source_neuro} to {target_neuro} in module {target_module}")
                    self._connections[source_neuro, target_neuro] = 1
    

    def _set_params(self, inhibitory_offset):
        """ 
        Defines a,b,c,d parameters of each Izenkevich neuron based on the type of neuron based the Lab 2 parameters.
        Inputs:
            - inhibitory_offset -- Index of first inhibitory neuron in the network
        Output:
            - (a,b,c,d) -- Tuple of a,b,c,d neuron parameters
        """
        r = np.random.rand(self._num_neurons)

        a = np.zeros(self._num_neurons)
        a[inhibitory_offset:] = 0.02 + (0.08 * r[inhibitory_offset:]) # inhibitory
        a[:inhibitory_offset] = 0.02 # excitatory 
        b = np.zeros(self._num_neurons)
        b[inhibitory_offset:] = 0.25 - (0.05 * r[inhibitory_offset:]) # inhibitory
        b[:inhibitory_offset] = 0.2 # excitatory 
        c = np.zeros(self._num_neurons)
        c[inhibitory_offset:] = -65 # inhibitory
        c[:inhibitory_offset] = -65 + (15 * (r[:inhibitory_offset] * r[:inhibitory_offset])) # excitatory 
        d = np.zeros(self._num_neurons)
        d[inhibitory_offset:] = 2 # inhibitory
        d[:inhibitory_offset] = 8 - (6 * (r[:inhibitory_offset] * r[:inhibitory_offset])) # excitatory 

        return (a,b,c,d)
    

    def _set_up_plotting(self):
        """
        Set up plotting parameters (only for aesthetic purposes).
        """
        plt.rcParams.update({
            "lines.linewidth": 1.5,
            "font.family": "serif",
            "font.size": 10,
            "pdf.fonttype":42})


    def assert_connections(self, neuro_per_module):
        """ 
        Checks if all the network connections (before rewiring) are valid based on the Courework instructions. 
        It checks that:
            - each module has exactly 1000 connections
            - each module has at most 4 outgoing connections to an inhibitory neuron
            - each inhibitory neuron has 4 incoming excitatory connections
            - each inhibitory neuron has connections to every other neuron in the network
            - there are no self-connections for every neuron
        Inputs:
            - neuro_per_module -- Number of (Excitatory) Neurons in each Module
        """
        # based on coursework specificatinos
        intra_edges = 1000
        inhib_offset = 800
        exc_to_inhib = 4
        inhib_to_all = intra_edges-1

        for neuro in range(self._num_neurons):
            module = neuro // neuro_per_module
            if neuro in self._excitatory:
                # assert np.sum(self._connections[neuro, :]) <= intra_edges+exc_to_inhib, f"Too many connections for neuron {neuro}"
                assert np.sum(self._communities[module, :, :]) == intra_edges, f"Number of community connections in community {module} not equal to {intra_edges}"
                assert np.sum(self._communities[module, :, inhib_offset:]) <= 4, f"Too many connections from community {module} to inhibitory neurons"
            elif neuro in self._inhibitory:
                assert np.count_nonzero((self._connections[:inhib_offset, neuro])) == exc_to_inhib, f"Number of incoming excitatory connections for neuron {neuro} not equal to {exc_to_inhib}"
                assert np.count_nonzero((self._connections[neuro, :])) == inhib_to_all, f"Number of outgoing inhibitory connections for neuron {neuro} not equal to {inhib_to_all}"
            assert (self._connections[neuro, neuro] == 0), f"Self connection for neuron {neuro}"
        print(f"All Network Connections are Valid.")


    def simulate_net(self, T=1000):
        """ 
        Simulate the network for T milliseconds and store the voltage and firing events for each neuron.
        Inputs:
            - T -- Duration of simulation (Defaults to 1000ms)
        """
        V = np.zeros((T, self._num_neurons))
        # self._fired = np.zeros((T, self._num_neurons))
        for t in range(T):
            I = np.zeros(self._num_neurons)
            # inject extra current I=15 based on a poisson distribution with lambda = 0.01
            I = np.random.poisson(0.01, self._num_neurons)
            I[np.argwhere(I>0)] += 15
            self._net.setCurrent(I)
            fired_ids = self._net.update()
            V[t,:], _ = self._net.getState()
            for id in fired_ids:
                self._fired[t, id] = 1


    def plot_weights(self, title):
        """ 
        Task a: Plots the weight matrix as a heatmap. Each dot is a connection between neuron x and neuron y
                on the x,y axis. 
        Inputs:
            - title -- Title of Plot
        """
        # visualize w as heatmap
        N = self._num_neurons
        neuro_per_module = len(self._excitatory)//self._num_modules
        plt.figure(figsize=(8,8))
        sns.heatmap(self._W != 0, cbar=False, cmap='gray')
        # sns.heatmap(W_p[p], cmap='gray')
        plt.xticks(np.arange(0, N, neuro_per_module), np.arange(0, N, neuro_per_module))
        plt.yticks(np.arange(0, N, neuro_per_module), np.arange(0, N, neuro_per_module))
        # plt.title(title)
        # plt.savefig(f'img/weights/weights_p{self._p}.png', dpi=300)
        plt.show()


    def plot_raster(self, T=1000):
        """
        Task b: plot the firing behaviour of each neuron during simulation. Each neuron spike is represented 
                as a blue dot at the time t of firing.
        Inputs:
            - T -- Duration of simulation (Defaults to 1000ms)
        """
        inhibitory_offset = len(self._excitatory) # to only plot the excitatory neurons
        neuron_events = []
        for neuron in range(self._fired.shape[1]):
            firing_times = np.where(self._fired[:, neuron] > 0)[0]  # Time indices where neuron has fired
            neuron_events.append(firing_times)
        neuron_indices = []
        firing_times_flat = []
        for neuron, times in enumerate(neuron_events):
            if neuron < inhibitory_offset:
                neuron_indices.extend([neuron] * len(times))
                firing_times_flat.extend(times)

        # Plot the raster plot 
        plt.figure(figsize=(12, 4))
        plt.scatter(firing_times_flat, neuron_indices, color='blue', s=10, marker='o')
        plt.xlabel('$Time (s^{-3})$')
        plt.ylabel('Neuron Number')
        plt.title(f'Raster plot for Network with p={self._p}')
        plt.xlim(0, T) 
        plt.ylim(inhibitory_offset, 0)  
        plt.tight_layout()
        # plt.savefig(f'img/raster/raster_p{self._p}.pdf', dpi=300)
        plt.show()


    def plot_rolling_mean_per_module(self, T=1000, window=50, shift=20):
        """
        Task c: plot the rolling mean for each module with a window of 50ms. Firing rates are downsampled 
                to obtain the mean by computing the average number of firings in 50ms windows shifted 
                every 20ms (Yields 50 data points for each module).
        Inputs:
            - T -- Duration of simulation (Defaults to 1000ms)
            - window -- window size for rolling mean (Defaults to 50ms)
            - shift -- shift size for rolling mean (Defaults to 20ms)
        """
        neuro_per_module = len(self._excitatory)//self._num_modules
        time_array = np.arange(0, T, shift)
        plt.figure(figsize=(15, 5))
        for module in range(self._num_modules):
            offset_start = neuro_per_module * module
            offset_end = neuro_per_module * (module+1)
            # Sum fired over the wole module
            fired_module = np.sum(self._fired[:, offset_start: offset_end], axis=1)
            rolling_mean = np.convolve(fired_module, np.ones(window) / window, mode='valid')[::shift]
            # Plot rolling mean
            plt.plot(time_array[:len(rolling_mean)], rolling_mean, label=f'Module {module + 1}', linewidth=0.6)

        plt.xlabel('Time (ms)')
        plt.ylabel('Mean Firing Rate $(spike \\times ms^{-1})$')
        plt.title(f'Rolling Mean Firing Rate for each module in Net with p={self._p}')
        plt.legend()
        # plt.savefig(f'img/firing/rolling_mean_p{self._p}.pdf', dpi=300)
        plt.show()