import networkx as nx
import numpy as np


class FactorGraph():
    """
    bi partite graph
    """

    def __init__(self):
        self.g = nx.Graph()
        self.names_var_nodes = []
        self.names_fact_nodes = []

    def add_variable_node(self, name, cardinality=None, distrib=None):
        """
        Add a variable node to the graph
        
        Arguments:
            name {str}              -- name of the variable
            cardinality {int}       -- number of possible values of the variable
            distrib {1d np.array}   -- distribution of the variable
        """
        if name in self.names_fact_nodes or name in self.names_var_nodes:
            raise ValueError("node {} already exists".format(name))

        if cardinality is None and distrib is None:
            raise ValueError("cardinality or distrib must be provided")
        
        if distrib is not None:
            distrib = np.array(distrib)
            if distrib.ndim != 1:
                raise ValueError("distrib must be a 1d array")            
            if cardinality is None:
                cardinality = len(distrib)
            elif cardinality != len(distrib):
                raise ValueError("distrib must have the same length as cardinality")
        else : 
            distrib = np.ones(cardinality)

        self.g.add_node(name, type='variable', cardinality=cardinality, distrib=distrib)
        self.names_var_nodes.append(name)


    def add_factor_node(self, name, variables, distrib):
        """
        Add a factor node to the graph

        Arguments:
            name {str}              -- name of the factor
            variables {list of str} -- list of variable names
            distrib {np.array}     -- distrib of the factor (!if provieded shape must mach with (card_var0, card_var1,...) !)
        """
        if name in self.names_fact_nodes or name in self.names_var_nodes:
            raise ValueError("node {} already exists".format(name))
        for var in variables :
            if var not in self.names_var_nodes:
                raise ValueError("factor node on unknown variable node")

        distrib = np.array(distrib)
        if distrib.ndim != len(variables):
            raise ValueError("distrib must be a {}d array (because you provided {} variables)".format(len(variables), len(variables)))
        
        needed_distrib_shape = tuple([self.g.nodes[var]['cardinality'] for var in variables])
        if distrib.shape != needed_distrib_shape:
            raise ValueError("distrib shape {} mismatch with needed cardinalaties of specified variables {}".format(distrib.shape, needed_distrib_shape))


        self.g.add_node(name, type='factor', variables=variables, distrib=distrib)
        self.names_fact_nodes.append(name)

        for var in variables:
            self.g.add_edge(name, var)


    def draw(self, layout='kamada', pos=None):
        if layout == 'bipartite':
            pos = nx.bipartite_layout(self.g, self.names_fact_nodes)
        elif layout == 'spring':
            pos = nx.spring_layout(self.g, pos=pos)
        elif layout == 'kamada':
            pos = nx.kamada_kawai_layout(self.g, pos=pos)
        elif pos != None:
            raise ValueError("pos must be None if layout is not specified")
        
        # nx.draw(self.g, pos, with_labels=True)
        nx.draw_networkx(self.g, pos, nodelist=self.names_var_nodes, node_shape='o')
        nx.draw_networkx(self.g, pos, nodelist=self.names_fact_nodes, node_shape='s', node_color='grey')
        

def build_image_factor_graph(image, distrib_int_obs, distrib_clc_neigh) : 
    g = FactorGraph()
    n_states = distrib_int_obs.shape[0]

    image_shape = image.shape
    for i, j in np.ndindex(image_shape):
        obs = np.zeros(n_states) #+ .1
        obs[int(image[i, j])] = 1

        obs_name = 'obs({}, {})'.format(i, j)
        clc_name = 'cls({}, {})'.format(i, j)

        g.add_variable_node(obs_name, n_states, obs)
        g.add_variable_node(clc_name, 2)
        
        g.add_factor_node('int_obs({},{})'.format(i, j), [obs_name, clc_name], distrib_int_obs)

    for i, j in np.ndindex(image_shape):
        if i+1 < image_shape[0]:
            g.add_factor_node('{}\n{}'.format((i, j), (i+1, j)), ['cls({}, {})'.format(i, j), 'cls({}, {})'.format(i+1, j)], distrib_clc_neigh)
        if j+1 < image_shape[1]:   
            g.add_factor_node('{}\n{}'.format((i, j), (i, j+1)), ['cls({}, {})'.format(i, j), 'cls({}, {})'.format(i, j+1)], distrib_clc_neigh)

    return g



class BP():
    def __init__(self, model:FactorGraph, debug=False):
        self.msg = {}
        self.model = model

    def belief(self, v_name) : 
        if v_name not in self.model.names_var_nodes:
            raise ValueError("unknown variable node")
        
        in_msgs = []
        for f_name in self.model.g.neighbors(v_name):
            in_msgs.append(self.get_fact2var_msg(f_name, v_name))

        distrib = np.array(in_msgs).prod(axis=0)
        distrib /= distrib.sum()

        return distrib

    def get_var2fact_msg(self, var, fact) :
        key = (var, fact)
        if key not in self.msg:
            self.msg[key] = self._compute_var2fact_msg(var, fact)

        return self.msg[key]

    def get_fact2var_msg(self, fact, var) :
        key = (fact, var)
        if key not in self.msg:
            self.msg[key] = self._compute_fact2var_msg(fact, var)

        return self.msg[key]
    
    
    def _compute_var2fact_msg(self, var, fact) :
        in_msgs = []
        for f_name in self.model.g.neighbors(var):
            if f_name != fact:
                in_msgs.append(self.get_fact2var_msg(f_name, var))
        
        if len(in_msgs) == 0:
            distrib = self.model.g.nodes[var]['distrib']
        else:
            distrib = np.array(in_msgs).prod(axis=0)

        distrib /= distrib.sum()
        return distrib


        
    def _compute_fact2var_msg(self, fact, var) :
        distrib = self.model.g.nodes[fact]['distrib']
        linked_vars = self.model.g.nodes[fact]['variables']

        for v_name in linked_vars[::-1]:
            if v_name != var:
                in_msg = self.get_var2fact_msg(v_name, fact)
                distrib = (distrib * in_msg).sum(axis=-1)

            else :
                distrib = np.moveaxis(distrib, -1, 0)   

        distrib /= distrib.sum() 
        return distrib
    

class Loopy_BP(BP):
    def __init__(self, model:FactorGraph):
        super().__init__(model)
        # self.model = model
        # self.msg = {}
        self.msg_new = {}
        self.t = 0
        self.init_msg()
    
    
    def get_fact2var_msg(self, fact, var) :
        return self.msg[(fact, var)]
    
    def get_var2fact_msg(self, var, fact) :
        return self.msg[(var, fact)]

    def loop(self):
        edges = np.array(list(self.model.g.edges))
        np.random.shuffle(edges)
        
        for n1, n2 in edges:
            name1 = n1 if self.model.g.nodes[n1]['type'] == 'variable' else n2
            name2 = n2 if n1 == name1 else n1

            self.msg_new[(name1, name2)] = self._compute_var2fact_msg(name1, name2)
            self.msg_new[(name2, name1)] = self._compute_fact2var_msg(name2, name1)
        
        self.msg.update(self.msg_new)
        self.t += 1


    def init_msg(self):
        for n1, n2 in self.model.g.edges:
            name1 = n1 if self.model.g.nodes[n1]['type'] == 'variable' else n2
            name2 = n2 if n1 == name1 else n1

            self.msg[(name1, name2)] = np.ones(self.model.g.nodes[name1]['cardinality'])
            self.msg[(name2, name1)] = np.ones(self.model.g.nodes[name1]['cardinality'])

            self.msg_new[(name1, name2)] = 0
            self.msg_new[(name2, name1)] = 0


class URW_BP(Loopy_BP):
    def __init__(self, model:FactorGraph, rho=.5):
        super().__init__(model)
        # self.model = model
        # self.msg = {}
        # self.msg_new = {}
        # self.t = 0
        # self.init_msg()

        self.rho = rho


    def _compute_var2fact_msg(self, var, fact) :
        in_msgs = []
        for f_name in self.model.g.neighbors(var):
            if f_name != fact:
                msg = self.get_fact2var_msg(f_name, var)
                msg = np.power(msg, self.rho)
                in_msgs.append(msg)
        
        if len(in_msgs) == 0:
            distrib = self.model.g.nodes[var]['distrib']
        else:
            distrib = np.array(in_msgs).prod(axis=0)

        msg = self.get_fact2var_msg(fact, var)
        distrib = distrib * np.power(msg, self.rho-1)

        distrib /= distrib.sum()
        return distrib


        
    def _compute_fact2var_msg(self, fact, var) :
        distrib = self.model.g.nodes[fact]['distrib']
        distrib = np.power(distrib, 1./self.rho)

        linked_vars = self.model.g.nodes[fact]['variables']

        for v_name in linked_vars[::-1]:
            if v_name != var:
                in_msg = self.get_var2fact_msg(v_name, fact)
                distrib = (distrib * in_msg).sum(axis=-1)

            else :
                distrib = np.moveaxis(distrib, -1, 0)   

        distrib /= distrib.sum() 
        return distrib