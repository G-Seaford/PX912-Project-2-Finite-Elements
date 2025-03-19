#%%
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshpy.triangle as mp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial.legendre import leggauss

#%%
class Dunavant:
    
    """
    A class that encapsulates the Dunavant quadrature rule for the reference triangle.
    Modified from C++ code by John Burkardt.
    
    Author : Gianluca Seaford
    Date   : 2021-03-16
    
    Reference:

        David Dunavant,
        High Degree Efficient Symmetrical Gaussian Quadrature Rules
        for the Triangle,
        International Journal for Numerical Methods in Engineering,
        Volume 21, 1985, pages 1129-1148.

        James Lyness, Dennis Jespersen,
        Moderate Degree Symmetric Quadrature Rules for the Triangle,
        Journal of the Institute of Mathematics and its Applications,
        Volume 15, Number 1, February 1975, pages 19-32.
        
        John Burkardt,
        triangle_dunavant_rule.cpp
        https://people.sc.fsu.edu/~jburkardt/cpp_src/triangle_dunavant_rule/triangle_dunavant_rule.html
        Accessed 2021-03-16.
        
    
    Attributes:
        rule      : integer, the chosen rule (from 1 to 20)
        degree    : polynomial degree of exactness (here taken equal to the rule number)
        order     : total number of quadrature points (computed from the suborder data)
        points    : NumPy array of shape (order, 2) with the quadrature points (in barycentric coordinates,
                  where we interpret the first two coordinates as the x and y of the point)
        weights   : NumPy array of length order containing the quadrature weights.
      
    """
    
    # Compressed suborder information
    _suborder = {
    1: [1],
    2: [3],
    3: [1, 3],
    4: [3, 3],
    5: [1, 3, 3],
    6: [3, 3, 6],
    7: [1, 3, 3, 6],
    8: [1, 3, 3, 3, 6],
    9: [1, 3, 3, 3, 3, 6],
    10: [1, 3, 3, 6, 6, 6],
    11: [3, 3, 3, 3, 3, 6, 6],
    12: [3, 3, 3, 3, 3, 6, 6, 6],
    13: [1, 3, 3, 3, 3, 3, 3, 6, 6, 6],
    14: [3, 3, 3, 3, 3, 3, 6, 6, 6, 6],
    15: [3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6],
    16: [1, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6],
    17: [1, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6],
    18: [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6],
    19: [1, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6],
    20: [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6]
    }
    # Compressed subrule data.
    # Each rule maps to a tuple (suborder_xyz, suborder_w), where:
    
    _subrules ={
        1: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333],
            [1.000000000000000]
        ),
        2: (
            [0.666666666666667, 0.166666666666667, 0.166666666666667],
            [0.333333333333333]
        ),
        3: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
             0.600000000000000, 0.200000000000000, 0.200000000000000],
            [-0.562500000000000, 0.520833333333333]
        ),
        4: (
            [0.108103018168070, 0.445948490915965, 0.445948490915965,
             0.816847572980459, 0.091576213509771, 0.091576213509771],
            [0.223381589678011, 0.109951743655322]
        ),
        5: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
             0.059715871789770, 0.470142064105115, 0.470142064105115,
             0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.225000000000000, 0.132394152788506, 0.125939180544827]
        ),
        6: (
            [0.501426509658179, 0.249286745170910, 0.249286745170910,
             0.873821971016996, 0.063089014491502, 0.063089014491502,
             0.053145049844817, 0.310352451033784, 0.636502499121399],
            [0.116786275726379, 0.050844906370207, 0.082851075618374]
        ),
        7: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
             0.479308067841920, 0.260345966079040, 0.260345966079040,
             0.869739794195568, 0.065130102902216, 0.065130102902216,
             0.048690315425316, 0.312865496004874, 0.638444188569810],
            [-0.149570044467682, 0.175615257433208, 0.053347235608838, 
             0.077113760890257]
        ),
        8: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
             0.081414823414554, 0.459292588292723, 0.459292588292723,
             0.658861384496480, 0.170569307751760, 0.170569307751760,
             0.898905543365938, 0.050547228317031, 0.050547228317031,
             0.008394777409958, 0.263112829634638, 0.728492392955404],
            [0.144315607677787, 0.095091634267285, 0.103217370534718,
             0.032458497623198, 0.027230314174435]
        ),
        9: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
             0.020634961602525, 0.489682519198738, 0.489682519198738,
             0.125820817014127, 0.437089591492937, 0.437089591492937,
             0.623592928761935, 0.188203535619033, 0.188203535619033,
             0.910540973211095, 0.044729513394453, 0.044729513394453,
             0.036838412054736, 0.221962989160766, 0.741198598784498],
            [0.097135796282799, 0.031334700227139, 0.077827541004774,
             0.079647738927210, 0.025577675658698, 0.043283539377289]
        ),
        10: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
             0.028844733232685, 0.485577633383657, 0.485577633383657,
             0.781036849029926, 0.109481575485037, 0.109481575485037,
             0.141707219414880, 0.307939838764121, 0.550352941820999,
             0.025003534762686, 0.246672560639903, 0.728323904597411,
             0.009540815400299, 0.066803251012200, 0.923655933587500],
            [0.090817990382754, 0.036725957756467, 0.045321059435528,
             0.072757916845420, 0.028327242531057, 0.009421666963733]
        ),
        11: (
            [-0.069222096541517, 0.534611048270758, 0.534611048270758,
             0.202061394068290, 0.398969302965855, 0.398969302965855,
             0.593380199137435, 0.203309900431282, 0.203309900431282,
             0.761298175434837, 0.119350912282581, 0.119350912282581,
             0.935270103777448, 0.032364948111276, 0.032364948111276,
             0.050178138310495, 0.356620648261293, 0.593201213428213,
             0.021022016536166, 0.171488980304042, 0.807489003159792],
            [0.000927006328961, 0.077149534914813, 0.059322977380774,
             0.036184540503418, 0.013659731002678, 0.052337111962204,
             0.020707659639141]
        ),
        12: (
            [0.023565220452390, 0.488217389773805, 0.488217389773805,
             0.120551215411079, 0.439724392294460, 0.439724392294460,
             0.457579229975768, 0.271210385012116, 0.271210385012116,
             0.744847708916828, 0.127576145541586, 0.127576145541586,
             0.957365299093579, 0.021317350453210, 0.021317350453210,
             0.115343494534698, 0.275713269685514, 0.608943235779788,
             0.022838332222257, 0.281325580989940, 0.695836086787803,
             0.025734050548330, 0.116251915907597, 0.858014033544073],
            [0.025731066440455, 0.043692544538038, 0.062858224217885,
             0.034796112930709, 0.006166261051559, 0.040371557766381,
             0.022356773202303, 0.017316231108659]
        ),
        13: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
             0.009903630120591, 0.495048184939705, 0.495048184939705,
             0.062566729780852, 0.468716635109574, 0.468716635109574,
             0.170957326397447, 0.414521336801277, 0.414521336801277,
             0.541200855914337, 0.229399572042831, 0.229399572042831,
             0.771151009607340, 0.114424495196330, 0.114424495196330,
             0.950377217273082, 0.024811391363459, 0.024811391363459,
             0.094853828379579, 0.268794997058761, 0.636351174561660,
             0.018100773278807, 0.291730066734288, 0.690169159986905,
             0.022233076674090, 0.126357385491669, 0.851409537834241],
            [0.052520923400802, 0.011280145209330, 0.031423518362454,
             0.047072502504194, 0.047363586536355, 0.031167529045794,
             0.007975771465074, 0.036848402728732, 0.017401463303822,
             0.015521786839045]
        ),
        14: (
            [0.022072179275643, 0.488963910362179, 0.488963910362179,
             0.164710561319092, 0.417644719340454, 0.417644719340454,
             0.453044943382323, 0.273477528308839, 0.273477528308839,
             0.645588935174913, 0.177205532412543, 0.177205532412543,
             0.876400233818255, 0.061799883090873, 0.061799883090873,
             0.961218077502598, 0.019390961248701, 0.019390961248701,
             0.057124757403648, 0.172266687821356, 0.770608554774996,
             0.092916249356972, 0.336861459796345, 0.570222290846683,
             0.014646950055654, 0.298372882136258, 0.686980167808088,
             0.001268330932872, 0.118974497696957, 0.879757171370171],
            [0.021883581369429, 0.032788353544125, 0.051774104507292,
             0.042162588736993, 0.014433699669777, 0.004923403602400,
             0.024665753212564, 0.038571510787061, 0.014436308113534,
             0.005010228838501]
        ),
        15: (
            [-0.013945833716486, 0.506972916858243, 0.506972916858243,
             0.137187291433955, 0.431406354283023, 0.431406354283023,
             0.444612710305711, 0.277693644847144, 0.277693644847144,
             0.747070217917492, 0.126464891041254, 0.126464891041254,
             0.858383228050628, 0.070808385974686, 0.070808385974686,
             0.962069659517853, 0.018965170241073, 0.018965170241073,
             0.133734161966621, 0.261311371140087, 0.604954466893291,
             0.036366677396917, 0.388046767090269, 0.575586555512814,
            -0.010174883126571, 0.285712220049916, 0.724462663076655,
             0.036843869875878, 0.215599664072284, 0.747556466051838,
             0.012459809331199, 0.103575616576386, 0.883964574092416],
            [0.001916875642849, 0.044249027271145, 0.051186548718852,
             0.023687735870688, 0.013289775690021, 0.004748916608192,
             0.038550072599593, 0.027215814320624, 0.002182077366797,
             0.021505319847731, 0.007673942631049]
        ),
        16: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
             0.005238916103123, 0.497380541948438, 0.497380541948438,
             0.173061122901295, 0.413469438549352, 0.413469438549352,
             0.059082801866017, 0.470458599066991, 0.470458599066991,
             0.518892500060958, 0.240553749969521, 0.240553749969521,
             0.704068411554854, 0.147965794222573, 0.147965794222573,
             0.849069624685052, 0.075465187657474, 0.075465187657474,
             0.966807194753950, 0.016596402623025, 0.016596402623025,
             0.103575692245252, 0.296555596579887, 0.599868711174861,
             0.020083411655416, 0.337723063403079, 0.642193524941505,
            -0.004341002614139, 0.204748281642812, 0.799592720971327,
             0.041941786468010, 0.189358492130623, 0.768699721401368,
             0.014317320230681, 0.085283615682657, 0.900399064086661],
            [0.046875697427642, 0.006405878578585, 0.041710296739387,
             0.026891484250064, 0.042132522761650, 0.030000266842773,
             0.014200098925024, 0.003582462351273, 0.032773147460627,
             0.015298306248441, 0.002386244192839, 0.019084792755899,
             0.006850054546542]
        ),
        17: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
             0.005658918886452, 0.497170540556774, 0.497170540556774,
             0.035647354750751, 0.482176322624625, 0.482176322624625,
             0.099520061958437, 0.450239969020782, 0.450239969020782,
             0.199467521245206, 0.400266239377397, 0.400266239377397,
             0.495717464058095, 0.252141267970953, 0.252141267970953,
             0.675905990683077, 0.162047004658461, 0.162047004658461,
             0.848248235478508, 0.075875882260746, 0.075875882260746,
             0.968690546064356, 0.015654726967822, 0.015654726967822,
             0.010186928826919, 0.334319867363658, 0.655493203809423,
             0.135440871671036, 0.292221537796944, 0.572337590532020,
             0.054423924290583, 0.319574885423190, 0.626001190286228,
             0.012868560833637, 0.190704224192292, 0.796427214974071,
             0.067165782413524, 0.180483211648746, 0.752351005937729,
             0.014663182224828, 0.080711313679564, 0.904625504095608],
            [0.033437199290803, 0.005093415440507, 0.014670864527638,
             0.024350878353672, 0.031107550868969, 0.031257111218620,
             0.024815654339665, 0.014056073070557, 0.003194676173779,
             0.008119655318993, 0.026805742283163, 0.018459993210822,
             0.008476868534328, 0.018292796770025, 0.006665632004165]
        ),
        18: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
             0.013310382738157, 0.493344808630921, 0.493344808630921,
             0.061578811516086, 0.469210594241957, 0.469210594241957,
             0.127437208225989, 0.436281395887006, 0.436281395887006,
             0.210307658653168, 0.394846170673416, 0.394846170673416,
             0.500410862393686, 0.249794568803157, 0.249794568803157,
             0.677135612512315, 0.161432193743843, 0.161432193743843,
             0.846803545029257, 0.076598227485371, 0.076598227485371,
             0.951495121293100, 0.024252439353450, 0.024252439353450,
             0.913707265566071, 0.043146367216965, 0.043146367216965,
             0.008430536202420, 0.358911494940944, 0.632657968856636,
             0.131186551737188, 0.294402476751957, 0.574410971510855,
             0.050203151565675, 0.325017801641814, 0.624779046792512,
             0.066329263810916, 0.184737559666046, 0.748933176523037,
             0.011996194566236, 0.218796800013321, 0.769207005420443,
             0.014858100590125, 0.101179597136408, 0.883962302273467,
            -0.035222015287949, 0.020874755282586, 1.014347260005363],
            [0.030809939937647, 0.009072436679404, 0.018761316939594,
             0.019441097985477, 0.027753948610810, 0.032256225351457,
             0.025074032616922, 0.015271927971832, 0.006793922022963,
            -0.002223098729920, 0.006331914076406, 0.027257538049138,
             0.017676785649465, 0.018379484638070, 0.008104732808192,
             0.007634129070725, 0.000046187660794]
        ),
        19: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
             0.020780025853987, 0.489609987073006, 0.489609987073006,
             0.090926214604215, 0.454536892697893, 0.454536892697893,
             0.197166638701138, 0.401416680649431, 0.401416680649431,
             0.488896691193805, 0.255551654403098, 0.255551654403098,
             0.645844115695741, 0.177077942152130, 0.177077942152130,
             0.779877893544096, 0.110061053227952, 0.110061053227952,
             0.888942751496321, 0.055528624251840, 0.055528624251840,
             0.974756272445543, 0.012621863777229, 0.012621863777229,
             0.003611417848412, 0.395754787356943, 0.600633794794645,
             0.134466754530780, 0.307929983880436, 0.557603261588784,
             0.014446025776115, 0.264566948406520, 0.720987025817365,
             0.046933578838178, 0.358539352205951, 0.594527068955871,
             0.002861120350567, 0.157807405968595, 0.839331473680839,
             0.223861424097916, 0.075050596975911, 0.701087978926173,
             0.034647074816760, 0.142421601113383, 0.822931324069857,
             0.010161119296278, 0.065494628082938, 0.924344252620784],
            [0.032906331388919, 0.010330731891272, 0.022387247263016,
             0.030266125869468, 0.030490967802198, 0.024159212741641,
             0.016050803586801, 0.008084580261784, 0.002079362027485,
             0.003884876904981, 0.025574160612022, 0.008880903573338,
             0.016124546761731, 0.002491941817491, 0.018242840118951,
             0.010258563736199, 0.003799928855302]
        ),
        20: (
            [0.333333333333333, 0.333333333333333, 0.333333333333333,
            -0.001900928704400, 0.500950464352200, 0.500950464352200,
             0.023574084130543, 0.488212957934729, 0.488212957934729,
             0.089726636099435, 0.455136681950283, 0.455136681950283,
             0.196007481363421, 0.401996259318289, 0.401996259318289,
             0.488214180481157, 0.255892909759421, 0.255892909759421,
             0.647023488009788, 0.176488255995106, 0.176488255995106,
             0.791658289326483, 0.104170855336758, 0.104170855336758,
             0.893862072318140, 0.053068963840930, 0.053068963840930,
             0.916762569607942, 0.041618715196029, 0.041618715196029,
             0.976836157186356, 0.011581921406822, 0.011581921406822,
             0.048741583664839, 0.344855770229001, 0.606402646106160,
             0.006314115948605, 0.377843269594854, 0.615842614456541,
             0.134316520547348, 0.306635479062357, 0.559048000390295,
             0.013973893962392, 0.249419362774742, 0.736606743262866,
             0.075549132909764, 0.212775724802802, 0.711675142287434,
            -0.008368153208227, 0.146965436053239, 0.861402717154987,
             0.026686063258714, 0.137726978828923, 0.835586957912363,
             0.010547719294141, 0.059696109149007, 0.929756171556853],
            [0.033057055541624, 0.000867019185663, 0.011660052716448,
             0.022876936356421, 0.030448982673938, 0.030624891725355,
             0.024368057676800, 0.015997432032024, 0.007698301815602,
            -0.000632060497488, 0.001751134301193, 0.016465839189576,
             0.004839033540485, 0.025804906534650, 0.008471091054441,
             0.018354914106280, 0.000704404677908, 0.010112684927462,
             0.003573909385950]
        )
    }
    
    def __init__(self, rule):
        """
        Initialise Dunavant quadrature rule.
        
        Args:
          rule (int): Rule index (must be between 1 and 20).
        """
        if not (1 <= rule <= 20):
            raise ValueError(f"Rule must be between 1 and 20; got {rule}.")
        self.rule = rule
        self.degree = self._compute_degree(rule)
        self.order = self._compute_order(rule)
        self.points, self.weights = self._compute_rule(rule)
        
    @staticmethod
    def _compute_degree(rule):
        return rule
    
    @classmethod
    def _compute_order(cls, rule):
        if rule not in cls._suborder:
            raise ValueError(f"Suborder data for rule {rule} not available.")
        return sum(cls._suborder[rule])
    
    @classmethod
    def _compute_rule(cls, rule):
        """
        Expand compressed subrule data into full quadrature points and weights.
        Returns:
          xy (ndarray): Array of shape (order, 2) containing quadrature points.
          w (ndarray): Array of length order with the weights.
        """
        if rule not in cls._suborder or rule not in cls._subrules:
            raise ValueError(f"Data for rule {rule} is not available.")
        suborder = cls._suborder[rule]
        suborder_xyz, suborder_w = cls._subrules[rule]
        suborder_num = len(suborder)
        total_order = sum(suborder)
        xy = np.zeros((total_order, 2))
        w = np.zeros(total_order)
        o = 0
        for s in range(suborder_num):
            if suborder[s] == 1:
                xy[o, 0] = suborder_xyz[3*s + 0]
                xy[o, 1] = suborder_xyz[3*s + 1]
                w[o] = suborder_w[s]
                o += 1
            elif suborder[s] == 3:
                for k in range(3):
                    idx = 3 * s + (k % 3)
                    idx_next = 3 * s + ((k + 1) % 3)
                    xy[o, 0] = suborder_xyz[idx]
                    xy[o, 1] = suborder_xyz[idx_next]
                    w[o] = suborder_w[s]
                    o += 1
            elif suborder[s] == 6:
                # First set of 3 points.
                for k in range(3):
                    idx = 3 * s + (k % 3)
                    idx_next = 3 * s + ((k + 1) % 3)
                    xy[o, 0] = suborder_xyz[idx]
                    xy[o, 1] = suborder_xyz[idx_next]
                    w[o] = suborder_w[s]
                    o += 1
                # Second set: reverse order.
                for k in range(3):
                    idx = 3 * s + ((k + 1) % 3)
                    idx_next = 3 * s + (k % 3)
                    xy[o, 0] = suborder_xyz[idx]
                    xy[o, 1] = suborder_xyz[idx_next]
                    w[o] = suborder_w[s]
                    o += 1
            else:
                raise ValueError(f"Illegal suborder value {suborder[s]} for rule {rule}.")
        return xy, w

#%%
class FiniteElement:
    
    def __init__(self):
        # Material properties
        self.Youngs_modulus = None
        self.Poisson_ratio = None
        self.plane_thickness = None
        
        # System properties
        self.domain = None
        self.tractions = None

        # Mesh properties        
        self.mesh = None
        self.nodes = None
        self.mesh_elements = None
        self.max_volume = None
        
        # System Matrices
        self.K = None
        self.F = None
        self.U = None
        
        # Stress and Strain Fields
        self.Stress = None
        self.Strain = None
        
        # Refinement History
        self.num_element_history = []
        self.error_history = []

        # Stress & Strain History
        self.stress_error_history = []
        self.strain_error_history = []
        
    @property
    def num_nodes(self):
        return self.nodes.shape[0]

    @property
    def num_elements(self):
        return self.mesh_elements.shape[0]
    
    def _set_system_params(self, domain, tractions, E, nu, thickness, init_vol):
        self.domain = domain
        self.tractions = tractions
        self.Youngs_modulus = E
        self.Poisson_ratio = nu
        self.plane_thickness = thickness
        self.max_volume = init_vol
        
    def _generate_mesh(self, max_volume=0.15):
        vertices = self.domain.get('vertices').tolist()
        segments = self.domain.get('edges').tolist()
        
        mesh_info = mp.MeshInfo()
        mesh_info.set_points(vertices)
        mesh_info.set_facets(segments)
        
        self.mesh = mp.build(mesh_info, max_volume=max_volume)
        self.nodes = np.array(self.mesh.points)
        self.mesh_elements = np.array(self.mesh.elements, dtype=int)
    
    def _shape_funcs(self, xi, eta):
        N = np.array([1 - xi - eta, xi, eta])
        dN = np.array([[-1, -1], [1, 0], [0, 1]])
        return N, dN
    
    def _constitutive_matrix(self):
        E = self.Youngs_modulus
        nu = self.Poisson_ratio
        return E / (1 - nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])
    
    def _apply_tractions(self):
        for key, tr in self.tractions.items():
            traction = tr['traction']
            nodes_edge = tr['nodes']
            n1, n2 = nodes_edge
            p1 = self.nodes[n1]
            p2 = self.nodes[n2]
            L = np.linalg.norm(p2 - p1)
            F_edge = (L / 2) * traction
            self.F[2*n1:2*n1+2] += F_edge
            self.F[2*n2:2*n2+2] += F_edge
    
    def _apply_boundary_conditions(self):
        fixed_dofs = []
        tol = 1e-6
        for i, coord in enumerate(self.nodes):
            if abs(coord[0]) < tol:
                fixed_dofs.extend([2*i, 2*i+1])
        for dof in fixed_dofs:
            self.K[dof, :] = 0
            self.K[:, dof] = 0
            self.K[dof, dof] = 1
            self.F[dof] = 0
            
    def _assemble_system(self):
        
        n_dof = 2 * self.num_nodes
    
        # Use LIL for efficient insertion
        K_lil = sp.lil_matrix((n_dof, n_dof))
        F = np.zeros(n_dof)
        C = self._constitutive_matrix()
    
        quad = Dunavant(3)
        det_tol = 1e-12
    
        for element in self.mesh_elements:
        
            n1, n2, n3 = element
            coords = self.nodes[[n1, n2, n3], :]
        
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            x3, y3 = coords[2]
        
            J = np.array([[x2 - x1, x3 - x1],
                        [y2 - y1, y3 - y1]])
            detJ = np.linalg.det(J)
        
            if detJ <= det_tol:
                if detJ <= 0:
                    raise ValueError("Non-positive element area encountered.")
                
                print("Warning: Skipping element with near-zero area.")
                continue  # skip only this element
        
            invJ = np.linalg.inv(J)
            Ke = np.zeros((6, 6))
        
            for qp, w_q in zip(quad.points, quad.weights):
                xi, eta = qp 
                N, dN_dxi = self._shape_funcs(xi, eta)
            
                dN_dx = dN_dxi @ invJ.T
                B = np.zeros((3, 6))
            
                for i in range(3):
                    B[0, 2*i]   = dN_dx[i, 0]
                    B[1, 2*i+1] = dN_dx[i, 1]
                    B[2, 2*i]   = dN_dx[i, 1]
                    B[2, 2*i+1] = dN_dx[i, 0]
                
                Ke += (B.T @ C @ B) * detJ * w_q
            
            # Collect global DOF indices
            dof_indices = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1]
        
            # Add to the global stiffness matrix
            for i in range(6):
                for j in range(6):
                    K_lil[dof_indices[i], dof_indices[j]] += Ke[i, j]
    
        # Convert to CSR once, after building
        self.K = K_lil.tocsr()
        self.F = F
        self._apply_tractions()
    
    def _solve_system(self):
        self.U = spla.spsolve(self.K, self.F)
    
    def _post_process(self):
        num_elems = self.mesh_elements.shape[0]
        
        self.Strain = np.zeros((num_elems, 3))
        self.Stress = np.zeros((num_elems, 3))
        
        C = self._constitutive_matrix()
        
        for idx, elem in enumerate(self.mesh_elements):
            n1, n2, n3 = elem
            coords = self.nodes[[n1, n2, n3], :]
            
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            x3, y3 = coords[2]
            
            J = np.array([[x2 - x1, x3 - x1],
                          [y2 - y1, y3 - y1]])
            
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            
            dN_dxi = np.array([[-1, -1],
                                [1, 0],
                                [0, 1]])
            
            dN_dx = dN_dxi @ invJ.T
            
            B = np.zeros((3, 6))
            for i in range(3):
                B[0, 2*i]   = dN_dx[i, 0]
                B[1, 2*i+1] = dN_dx[i, 1]
                B[2, 2*i]   = dN_dx[i, 1]
                B[2, 2*i+1] = dN_dx[i, 0]
                
            dof_indices = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1]
            
            Ue = self.U[dof_indices]
            
            strain = B @ Ue
            stress = C @ strain
            
            self.Strain[idx, :] = strain
            self.Stress[idx, :] = stress
    
    def compute_local_errors(self):
        """
        
        Returns:
          local_rel_stress_errs: Array of local relative stress errors.
          local_rel_strain_errs: Array of local relative strain errors.
        """
        num_nodes = self.nodes.shape[0]
    
        # Accumulate stresses and strains into nodal sums.
        node_stress_sum = [np.zeros(3) for _ in range(num_nodes)]
        node_strain_sum = [np.zeros(3) for _ in range(num_nodes)]
        node_count = [0 for _ in range(num_nodes)]
    
        for idx, elem in enumerate(self.mesh_elements):
            for n in elem:
                node_stress_sum[n] += self.Stress[idx]
                node_strain_sum[n] += self.Strain[idx]
                node_count[n] += 1
    
        # Compute recovered (averaged) nodal values.
        recovered_node_stress = np.array([
            node_stress_sum[i] / node_count[i] if node_count[i] > 0 else node_stress_sum[i]
            for i in range(num_nodes)])
        
        recovered_node_strain = np.array([
            node_strain_sum[i] / node_count[i] if node_count[i] > 0 else node_strain_sum[i]
            for i in range(num_nodes)])
    
        num_elems = self.mesh_elements.shape[0]

        local_rel_stress_errs = np.zeros(num_elems)
        local_rel_strain_errs = np.zeros(num_elems)
    
        # Loop over each element.
        for i, elem in enumerate(self.mesh_elements):
            nodes = list(elem)

            # Recovered average for this element
            rec_stress = np.mean(recovered_node_stress[nodes], axis=0)
            rec_strain = np.mean(recovered_node_strain[nodes], axis=0)

            # Actual computed (Gauss-point) stress/strain for this element
            elem_stress = self.Stress[i]
            elem_strain = self.Strain[i]

            # Compute difference vectors
            stress_diff = rec_stress - elem_stress
            strain_diff = rec_strain - elem_strain

            stress_rmse = np.sqrt(np.mean(stress_diff**2))
            strain_rmse = np.sqrt(np.mean(strain_diff**2))

            stress_norm = np.sqrt(np.mean(rec_stress**2))
            strain_norm = np.sqrt(np.mean(rec_strain**2))

            if stress_norm > 1e-12:
                local_rel_stress_errs[i] = stress_rmse #/ stress_norm
            else:
                local_rel_stress_errs[i] = 0.0

            if strain_norm > 1e-12:
                local_rel_strain_errs[i] = strain_rmse #/ strain_norm
            else:
                local_rel_strain_errs[i] = 0.0
    
        return local_rel_stress_errs, local_rel_strain_errs
    
    def compute_integration_error(self, f, rule_low=3, rule_high=20):
        errors = []
        centroids = []
        
        for element in self.mesh_elements:
            n1, n2, n3 = element
            coords = self.nodes[[n1, n2, n3], :]
            cx, cy = np.mean(coords, axis=0)
            centroids.append([cx, cy])
            
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            x3, y3 = coords[2]
            
            J = np.array([[x2-x1, x3-x1],
                          [y2-y1, y3-y1]])
            detJ = np.linalg.det(J)
            quad_low = Dunavant(rule_low)
            I_low = 0.0
            
            for qp, w_q in zip(quad_low.points, quad_low.weights):
                xi, eta = qp
                x_q = x1 + (x2-x1)*xi + (x3-x1)*eta
                y_q = y1 + (y2-y1)*xi + (y3-y1)*eta
                I_low += f(x_q, y_q) * w_q
                
            I_low *= detJ
            
            quad_high = Dunavant(rule_high)
            I_high = 0.0
            for qp, w_q in zip(quad_high.points, quad_high.weights):
                xi, eta = qp
                x_q = x1 + (x2-x1)*xi + (x3-x1)*eta
                y_q = y1 + (y2-y1)*xi + (y3-y1)*eta
                I_high += f(x_q, y_q) * w_q
            I_high *= detJ
            
            error = abs(I_high - I_low) / abs(I_high) * 100 if abs(I_high) > 1e-12 else 0.0
            errors.append(error)
            
        return np.array(centroids), np.array(errors)
    
    def local_refine(self, tol_stress, tol_strain):
        """
        Perform local (adaptive) refinement by subdividing only those elements whose
        stress norm exceeds the threshold.
        This implementation uses a simple midpoint bisection for each edge.
        """
        new_nodes = self.nodes.tolist()
        edge_to_mid = {} 
        new_elements = []
        
        def get_midpoint(n1, n2):
            key = tuple(sorted((n1, n2)))
            if key in edge_to_mid:
                return edge_to_mid[key]
            else:
                p1 = new_nodes[n1]
                p2 = new_nodes[n2]
                mid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
                new_nodes.append(mid)
                mid_index = len(new_nodes) - 1
                edge_to_mid[key] = mid_index
                return mid_index
        
         # Compute local relative errors first.
        local_rel_stress, local_rel_strain = self.compute_local_errors()
        
        for i, elem in enumerate(self.mesh_elements):
            
            # Refine if either relative stress error or relative strain error exceeds its tolerance.
            if local_rel_stress[i] > tol_stress or local_rel_strain[i] > tol_strain:
                n1, n2, n3 = elem
                m12 = get_midpoint(n1, n2)
                m23 = get_midpoint(n2, n3)
                m31 = get_midpoint(n3, n1)
                
                # Subdivide the triangle into 4 sub-triangles.
                new_elements.extend([
                    [n1, m12, m31],
                    [n2, m23, m12],
                    [n3, m31, m23],
                    [m12, m23, m31]])
                
            else:
                new_elements.append(list(elem))
        
        self.nodes = np.array(new_nodes)
        self.mesh_elements = np.array(new_elements, dtype=int) §
        print(f"Local refinement: now {len(new_elements)} elements, {self.num_nodes} nodes.")
    
    def run_analysis(self, tol_stress=1e-2, tol_strain=1e-2, max_iterations=20):
        """
        Main driver routine. In each iteration the system is assembled, solved and post‐processed.
        Convergence is checked using the maximum relative stress and strain errors. If either exceeds
        its tolerance, local refinement is triggered (only those elements with too high errors are subdivided).
        Convergence data (number of elements and global error) is stored.
        """
        iteration = 0
        self._generate_mesh()  # start with a coarse mesh
        
        while iteration < max_iterations:
            print(f"\nIteration {iteration+1}")
            
            self._assemble_system()
            self._apply_boundary_conditions()
            self._solve_system()
            self._post_process()
    
            local_rel_stress, local_rel_strain = self.compute_local_errors()
    
            max_rel_stress_error = np.max(local_rel_stress)
            max_rel_strain_error = np.max(local_rel_strain)
    
             # Store iteration data
            self.num_element_history.append(len(self.mesh_elements))
            self.stress_error_history.append(max_rel_stress_error/np.mean(local_rel_stress))
            self.strain_error_history.append(max_rel_strain_error)
    
            print(f"Maximum relative stress error : {max_rel_stress_error:.4e}")
            print(f"Maximum relative strain error : {max_rel_strain_error:.4e}")
    
            # Stop if both relative errors are below tolerance.
            if max_rel_stress_error < tol_stress and max_rel_strain_error < tol_strain:
                print("Error tolerances met. Refinement stopping.")
                break
    
            # Trigger local refinement on elements that exceed tolerance.
            self.local_refine(tol_stress, tol_strain)
    
            iteration += 1
        if iteration == max_iterations:
            print("Warning: Maximum iterations reached.")
        else:
            print("Analysis complete.")
            
        self._assemble_system()
        self._apply_boundary_conditions()
        self._solve_system()
        self._post_process()
        
        print("Final system solution computed on refined mesh.")
    
    def plot_mesh(self):
        triang = tri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], triangles=self.mesh_elements)
        plt.figure()
        plt.triplot(triang, 'k-', lw=0.5)
        plt.title('Finite Element Mesh')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def plot_stress_field(self, component=0):
        """
        Plot the stress field using nodal values (recovered by averaging the element stresses)
        and using Gouraud shading.
        """
        dims = ['x','y']
        
        # Compute nodal stress values by averaging over all elements sharing a node.
        num_nodes = self.nodes.shape[0]
        stress_sum = np.zeros(num_nodes)
        count = np.zeros(num_nodes)
        for i, elem in enumerate(self.mesh_elements):
            for n in elem:
                stress_sum[n] += self.Stress[i, component]
                count[n] += 1
                
        # Avoid division by zero.
        nodal_stress = stress_sum / np.maximum(count, 1)
    
        triang = tri.Triangulation(self.nodes[:, 0], self.nodes[:, 1],
                                triangles=self.mesh_elements)
    
        plt.figure()
        # Gouraud shading uses values at vertices.
        tpc = plt.tripcolor(triang, nodal_stress, edgecolors='k', shading='gouraud')
        plt.colorbar(tpc, label=f'Stress component (Pa)')
        plt.title(f'Stress Field in {dims[component]}-direction with Gouraud shading')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        
    def plot_strain_field(self, component=0):
        """
        Plot the strain field using nodal values (recovered by averaging the element stresses)
        and using Gouraud shading.
        """
        dims = ['x','y']
        
        # Compute nodal strain values by averaging over all elements sharing a node.
        num_nodes = self.nodes.shape[0]
        strain_sum = np.zeros(num_nodes)
        count = np.zeros(num_nodes)
        for i, elem in enumerate(self.mesh_elements):
            for n in elem:
                strain_sum[n] += self.Strain[i, component]
                count[n] += 1
                
        # Avoid division by zero.
        nodal_stress = strain_sum / np.maximum(count, 1)
    
        triang = tri.Triangulation(self.nodes[:, 0], self.nodes[:, 1],
                                triangles=self.mesh_elements)
    
        plt.figure()
        
        # Gouraud shading uses values at vertices.
        tpc = plt.tripcolor(triang, nodal_stress, edgecolors='k', shading='gouraud')
        plt.colorbar(tpc, label=f'Strain component')
        plt.title(f'Strain Field in {dims[component]}-direction with Gouraud shading')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def plot_convergence(self):
        """
        Plot the maximum relative stress and strain errors vs. the number of elements
        on a single log-log plot.
        """
        plt.figure()
    
        # Plot max stress error vs. number of elements
        plt.loglog(self.num_element_history, self.stress_error_history,
                   marker='o', label='Max Relative Stress Error')
    
        # Plot max strain error vs. number of elements
        plt.loglog(self.num_element_history, self.strain_error_history,
                   marker='s', label='Max Relative Strain Error')
    
        plt.title('Convergence History')
        plt.xlabel('Number of Elements')
        plt.ylabel('Relative Error')
        plt.grid(True, which='both', ls='--')
        plt.legend()
        plt.show()
    
    def plot_integration_error(self, f, rule_low=3, rule_high=20):
        centroids, errors = self.compute_integration_error(f, rule_low, rule_high)
        centroids = np.array(centroids)
        plt.figure()
        tcf = plt.tricontourf(centroids[:, 0], centroids[:, 1], errors, levels=14, cmap='viridis')
        plt.colorbar(tcf, label='Integration Error (%)')
        plt.title('Integration Error Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

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

fe_solver = FiniteElement()
fe_solver._set_system_params(domain, tractions, E, nu, thickness, 0.1)

fe_solver.run_analysis(tol_stress=1e-2, tol_strain=1e-2, max_iterations=5)

fe_solver.plot_mesh()

fe_solver.plot_stress_field(component=0)
fe_solver.plot_stress_field(component=1)

fe_solver.plot_strain_field(component=0)
fe_solver.plot_strain_field(component=1)

fe_solver.plot_convergence()

# %%
