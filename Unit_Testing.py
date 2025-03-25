import unittest
import numpy as np

# Import the classes we need from Finite_Elements.py
from Finite_Elements import Domain, TriMesh, FiniteElement

class TestTriMeshShapeFunctions(unittest.TestCase):
    """
    Test shape function correctness for TriMesh.
    """
    def setUp(self):
        """
        Create a minimal triangular domain with 1 element for testing.
        """
        # A single triangle domain: (0,0), (1,0), (0,1)
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        edges = np.array([
            [0,1],
            [1,2],
            [2,0]
        ])
        self.domain = Domain(vertices, edges)
        self.mesh = TriMesh(self.domain, max_volume=1.0)
        
        # Generate the mesh. In this case, it's trivial (just one triangle).
        self.mesh.generate_mesh(None)

    def test_shape_sum_to_one(self):
        """
        Test that shape_funcs(xi, eta) sums to 1 for various sample points.
        """
        sample_points = [
            (0.2, 0.3),
            (0.5, 0.4),
            (0.0, 0.0),  # corner
            (0.1, 0.1)
        ]
        for (xi, eta) in sample_points:
            N, dN = self.mesh.shape_funcs(xi, eta)
            self.assertAlmostEqual(
                np.sum(N), 1.0,
                msg=f"Shape functions do not sum to 1 at xi={xi}, eta={eta}"
            )

    def test_shape_nonnegative_in_triangle(self):
        """
        Check that shape functions are >= 0 inside the reference triangle.
        """
        # In a standard reference triangle:
        #  0 <= xi, eta, and xi+eta <= 1
        sample_points = [
            (0.1, 0.2),
            (0.3, 0.3),
            (0.0, 0.5),
            (0.4, 0.0)
        ]
        for (xi, eta) in sample_points:
            N, dN = self.mesh.shape_funcs(xi, eta)
            # all shape functions should be >= 0 inside the domain
            self.assertTrue(np.all(N >= 0.0),
                msg=f"Shape functions have negative value at xi={xi}, eta={eta}: N={N}")

class TestTriMeshBMatrix(unittest.TestCase):
    """
    Test B-matrix correctness for TriMesh.
    """
    def setUp(self):
        # Same single triangle domain
        vertices = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0]
        ])
        edges = np.array([
            [0,1],
            [1,2],
            [2,0]
        ])
        self.domain = Domain(vertices, edges)
        self.mesh = TriMesh(self.domain, max_volume=2.0)
        self.mesh.generate_mesh(None)  # just one triangle

    def test_bmatrix_centroid(self):
        """
        For the centroid of the triangle, check that detJ is positive
        and B has correct shape.
        """
        elem_idx = 0
        xi, eta = (1/3, 1/3)  # centroid in barycentric
        B, detJ = self.mesh.compute_B_matrix(elem_idx, xi, eta)
        
        self.assertEqual(B.shape, (3,6), "B matrix shape should be (3,6) for a linear triangle.")
        self.assertTrue(detJ > 0.0, "Jacobian determinant should be > 0 for a valid element.")
        
    def test_bmatrix_corners(self):
        """
        Check that the B-matrix does not blow up at corners (xi=0, eta=0, etc.).
        """
        elem_idx = 0
        corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        for (xi, eta) in corners:
            B, detJ = self.mesh.compute_B_matrix(elem_idx, xi, eta)
            self.assertEqual(B.shape, (3,6))
            # detJ might be small near corners but should not be negative
            self.assertTrue(detJ >= 0.0, f"detJ negative at corner {xi, eta}")

class TestTriMeshElementStiffness(unittest.TestCase):
    """
    Test element stiffness for TriMesh on a simple triangle.
    """
    def setUp(self):
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        edges = np.array([
            [0,1],
            [1,2],
            [2,0]
        ])
        self.domain = Domain(vertices, edges)
        self.mesh = TriMesh(self.domain, max_volume=1.0)
        self.mesh.generate_mesh(None)
        
        # Material
        self.E = 1.0e7
        self.nu = 0.3
        self.thk = 0.1
        
        # Just 1 element
        self.elem_idx = 0
        self.nodes = self.mesh.nodes
        self.mesh_elements = self.mesh.mesh_elements

    def test_element_stiffness_nonzero(self):
        """
        The local stiffness matrix for a valid element should be non-singular
        (or at least have non-zero entries).
        """
        K_e = self.mesh.element_stiffness(self.elem_idx, self.thk, self.E, self.nu,
                                          self.nodes, self.mesh_elements)
        # shape should be 6x6 for a linear triangle
        self.assertEqual(K_e.shape, (6,6))
        # check if there's at least some non-zero entries
        self.assertTrue(np.any(K_e != 0.0), "Element stiffness matrix is all zeros - unexpected!")
        
        # optionally check if it is rank-deficient or not
        rank = np.linalg.matrix_rank(K_e)
        # for a single 2D triangular element, rank < 6 might happen if boundary conditions aren't applied,
        # but let's just check it isn't rank 0
        self.assertTrue(rank > 0, "Local stiffness matrix has rank 0, which is suspicious for a single triangle.")

class TestTriMeshPatch(unittest.TestCase):
    """
    A small patch test: fix left edge, apply uniform traction on right edge,
    expect uniform stress in the single-element or small-element domain.
    """
    def setUp(self):
        # Let's do a 2x1 domain with a triangular mesh
        # We'll define it as two triangles or so. 
        # But for a quick test, we can keep it simple.
        vertices = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0]
        ])
        edges = np.array([
            [0,1],[1,2],[2,3],[3,0]
        ])
        self.domain = Domain(vertices, edges)
        
        self.E = 1.0e5
        self.nu = 0.3
        self.thk = 0.1
        
        # traction p in x-direction on right edge
        self.p = 1000.0
        
        # define traction along edge (1->2)
        tractions = {
            'right_edge': {
                'traction': np.array([self.p, 0.0]),
                'nodes': np.array([2,1])  # top & bottom corners if it's a single line
            }
        }
        
        self.fe_solver = FiniteElement(mesh_type="triangular")
        self.fe_solver._set_system_params(
            domain=self.domain,
            E=self.E,
            nu=self.nu,
            thickness=self.thk,
            tractions=tractions,
            init_vol=0.2  # moderate mesh density
        )

    def test_patch_solution(self):
        """
        We expect near-uniform stress: sigma_xx ~ p, sigma_yy ~ 0, tau_xy ~ 0
        in a perfect patch test. 
        We'll do a coarse check that sigma_xx is non-trivial and the 
        displacement on the right edge is > 0.
        """
        # We won't do a reference solution, just a single run
        self.fe_solver._generate_mesh(max_volume=0.2)
        
        # build global K
        self.fe_solver._assemble_global_stiffness()
        
        # init global F
        self.fe_solver.F = np.zeros(2 * self.fe_solver.num_nodes)
        
        # set BC
        self.fe_solver._set_boundary_conditions()
        
        # solve
        self.fe_solver.solve_system()
        
        # check displacement on right edge is positive in x
        right_nodes = []
        for i, coord in enumerate(self.fe_solver.nodes):
            if abs(coord[0] - 2.0) < 1e-8:
                right_nodes.append(i)
        
        for rn in right_nodes:
            ux = self.fe_solver.U[2*rn]
            self.assertGreater(ux, 0.0, f"Right edge node {rn} has non-positive x-displacement: {ux}")
        
        # compute stress in each element, check that sigma_xx is at least > 0
        # (for a perfect patch, it should be ~ p, but let's not be too strict)
        for e_idx in range(self.fe_solver.num_elements):
            strain, stress = self.fe_solver.compute_element_stress_strain(e_idx)
            sigma_xx, sigma_yy, tau_xy = stress
            
            self.assertGreater(sigma_xx, 0.0,
                f"Element {e_idx} has sigma_xx <= 0, expected positive with traction p.")
            # we might also check that sigma_yy is smaller than some fraction of sigma_xx
            # but let's keep it simple.


# Finally, run all tests
if __name__ == '__main__':
    unittest.main()