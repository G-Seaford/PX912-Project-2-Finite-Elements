import matplotlib.pyplot as plt
import meshpy as mp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial.legendre import leggauss


#%%
class FiniteElement:
    
    def __init__(self):
        
        # Material properties #
        self.Youngs_modulus = None
        self.Poisson_ratio = None
        self.plane_thickness = None
        
        # System properties #
        self.domain = None
        self.tractions = None

        # Mesh properties #        
        self.mesh = None
        self.nodes = None
        self.num_nodes = None
        self.mesh_elements = None
        
        # System Matrices #
        self.K = None
        self.F = None
        self.U = None
        
        # Stress and Strain Fields #
        self.Stress = None
        self.Strain = None
        
        self.error_stress = None
        self.error_strain = None
        
    def _set_system_params(self,domain,tractions,E,nu,thickness):
        """
        Set the system parameters

        Args:
            domain      (dict)  : Dictionary containing the domain vertices and edges
            tractions   (dict)  : Dictionary containing tractions
            E           (float) : Young's Modulus
            nu          (float) : Poisson's Ratio
            thickness   (float) : Out of plane thickness
        """
        
        self.domain = domain
        self.tractions = tractions
        self.Youngs_modulus = E
        self.Poisson_ratio = nu
        self.plane_thickness = thickness
        
    def _generate_mesh(self, max_volume=0.1):
        """
        Use Meshpy to generate triangular mesh

        Args:
            max_volume  (float, optional): Controls the maximum volume size for mesh elements. Defaults to 0.1.
        """
        
        vertices = self.domain.get('vertices').tolist()
        segments = self.domain.get('edges').tolist()
        
        mesh_info = mp.MeshInfo()
        mesh_info.set_points(vertices)
        mesh_info.set_facets(segments)
        
        self.mesh = mp.build(mesh_info,max_volume=max_volume)
        self.nodes = np.array(self.mesh.points)
        self.mesh_elements = np.array(self.mesh.elements, dtype=float)
        
        self.num_nodes = self.nodes.shape[0]
        
    def _shape_funcs(self, xi, eta):
        """
        Calculate the shape functions and their derivatives

        Args:
            xi (float): xi coordinate
            eta (float): eta coordinate

        Returns:
            N (np.array): Shape functions
            dN (np.array): Derivatives of shape functions
        """
        
        N = np.array([1-xi-eta, xi, eta])
        dN = np.array([[-1, -1], [1, 0], [0, 1]])

        return N, dN
    
    def _get_dunavant_quadrature(self, order=1):
    
#%%

E = 4.4e7 # Pa
nu = 0.37
thickness = 0.08 # m
alpha = 0.8 # m

domain = {
    'vertices': np.array(
                [[0.0,0.0],
                [2.0,0.5],
                [2.0,1.0],
                [alpha,1.0],
                [0.0, 1.0]
                ]),
    
    'edges':    np.array(
                [[0,1],
                [1,2],
                [2,3],
                [3,4],
                [4,0]
                ])
}
    
tractions = {
    't1': { # Traction t1 (146,260 kPa) acts on the edge between nodes 2 and 3.
        'traction'  :  np.array([146.0e3,260.0e3]), # Pa
        'nodes'     :   np.array([2,3])
        },
    
    't2' : {  # Traction t2 (1900,0 kPa) acts on the edge between nodes 1 and 2.
        'traction'  :   np.array([1900.0e3,0.0]), #Pa
        'nodes'     :   np.array([1,2])
        }
    }


# %%
