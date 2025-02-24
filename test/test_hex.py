import unittest
import numpy as np
import tensorflow as tf

from tfscripts.hex import conv
from tfscripts.hex import icecube
from tfscripts.hex import rotation


class TestHexKernels(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState(42)

    def test_hex_kernel(self):
        """Test HexKernel"""

        test_cases = [
            {
                "filter_size": [2, 0, 3, 2],
                "n_vars": 7,
                "kernel": np.array(
                    [
                        [
                            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                            [
                                [-1.4257220029830933, -1.0143787860870361],
                                [0.39387720823287964, -1.9629442691802979],
                                [-0.10814568400382996, 0.11988475173711777],
                            ],
                            [
                                [0.4779662489891052, -1.6896843910217285],
                                [0.4115297496318817, -0.6118844151496887],
                                [-0.8208955526351929, -1.7109564542770386],
                            ],
                        ],
                        [
                            [
                                [-0.019794760271906853, 0.40792471170425415],
                                [-0.11573483049869537, -0.30879053473472595],
                                [-0.9291547536849976, 0.2432696521282196],
                            ],
                            [
                                [-0.4925185739994049, 0.31435173749923706],
                                [-0.9397227168083191, -0.4897875487804413],
                                [-0.3880728483200073, 0.27374207973480225],
                            ],
                            [
                                [0.7232375741004944, 0.48548829555511475],
                                [0.5194052457809448, -0.5773778557777405],
                                [0.7949565052986145, 1.7222315073013306],
                            ],
                        ],
                        [
                            [
                                [-0.03933761268854141, 0.44203484058380127],
                                [-0.4911632537841797, -0.012474085204303265],
                                [0.18765877187252045, -1.074103832244873],
                            ],
                            [
                                [0.9184949398040771, -1.2781122922897339],
                                [0.4061416685581207, 0.4695652723312378],
                                [-1.096517562866211, 0.9345183968544006],
                            ],
                            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                        ],
                    ]
                ),
            },
            {
                "filter_size": [3, 1, 2],
                "n_vars": 25,
                "kernel": np.array(
                    [
                        [
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [-1.4257220029830933, -1.0143787860870361],
                            [0.0, 0.0],
                            [0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.4779662489891052, -1.6896843910217285],
                            [-0.019794760271906853, 0.40792471170425415],
                            [-0.4925185739994049, 0.31435173749923706],
                            [0.7232375741004944, 0.48548829555511475],
                        ],
                        [
                            [0.0, 0.0],
                            [-0.03933761268854141, 0.44203484058380127],
                            [0.9184949398040771, -1.2781122922897339],
                            [-0.6554914116859436, 0.7017678022384644],
                            [-1.0997180938720703, -0.7472116351127625],
                            [-0.8416294455528259, 0.2393561452627182],
                            [0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0],
                            [0.5353959202766418, -0.6356369256973267],
                            [0.011429687030613422, -1.0444294214248657],
                            [0.8294717073440552, -1.1644237041473389],
                            [-0.5676082372665405, 1.6229828596115112],
                            [-1.7003906965255737, -1.0814650058746338],
                            [0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0],
                            [0.5154496431350708, -0.6644119024276733],
                            [0.4813891053199768, -1.372441291809082],
                            [-1.166320562362671, -0.726149320602417],
                            [-0.06755267083644867, 1.1745551824569702],
                            [-0.39131826162338257, -0.7624845504760742],
                            [0.0, 0.0],
                        ],
                        [
                            [0.9889782667160034, -1.0488485097885132],
                            [0.09571779519319534, 0.8825100064277649],
                            [-0.209646537899971, -0.11976470053195953],
                            [1.8110568523406982, 0.26369625329971313],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.2035822570323944, -0.5185909867286682],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                        ],
                    ]
                ),
            },
            {
                "filter_size": [4, 0],
                "n_vars": 37,
                "kernel": np.array(
                    [
                        [
                            0.0,
                            0.0,
                            0.0,
                            -1.4257220029830933,
                            0.4779662489891052,
                            -0.019794760271906853,
                            -0.4925185739994049,
                        ],
                        [
                            0.0,
                            0.0,
                            0.7232375741004944,
                            -0.03933761268854141,
                            0.9184949398040771,
                            -0.6554914116859436,
                            -1.0997180938720703,
                        ],
                        [
                            0.0,
                            -0.8416294455528259,
                            0.5353959202766418,
                            0.011429687030613422,
                            0.8294717073440552,
                            -0.5676082372665405,
                            -1.7003906965255737,
                        ],
                        [
                            0.5154496431350708,
                            0.4813891053199768,
                            -1.166320562362671,
                            -0.06755267083644867,
                            -0.39131826162338257,
                            0.9889782667160034,
                            0.09571779519319534,
                        ],
                        [
                            -0.209646537899971,
                            1.8110568523406982,
                            0.2035822570323944,
                            1.2006571292877197,
                            -1.4237172603607178,
                            1.2166802883148193,
                            0.0,
                        ],
                        [
                            -0.5895527601242065,
                            -1.7983052730560303,
                            -0.7603991031646729,
                            -1.3157434463500977,
                            0.6037954688072205,
                            0.0,
                            0.0,
                        ],
                        [
                            -1.0926811695098877,
                            -1.2689539194107056,
                            -0.5327851176261902,
                            0.8136177659034729,
                            0.0,
                            0.0,
                            0.0,
                        ],
                    ]
                ),
            },
        ]

        for test in test_cases:
            tf.random.set_seed(42)
            kernel_obj = conv.HexKernel(
                filter_size=test["filter_size"],
                get_ones=False,
                float_precision="float32",
                seed=42,
                name="HexKernel",
            )
            kernel = kernel_obj()
            var_list = kernel_obj.var_list

            self.assertEqual(len(var_list), test["n_vars"])
            self.assertTrue(np.allclose(test["kernel"], kernel, atol=1e-6))

    def test_icecube_kernel(self):
        """Test IceCubeKernel"""

        test_cases = [
            {
                "filter_size": [1, 2],
                "n_vars": 78,
                "kernel": np.array(
                    [
                        [
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[-1.4257220029830933, -1.0143787860870361]],
                            [[0.4779662489891052, -1.6896843910217285]],
                            [[-0.019794760271906853, 0.40792471170425415]],
                            [[-0.4925185739994049, 0.31435173749923706]],
                            [[0.7232375741004944, 0.48548829555511475]],
                            [[-0.03933761268854141, 0.44203484058380127]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.9184949398040771, -1.2781122922897339]],
                            [[-0.6554914116859436, 0.7017678022384644]],
                            [[-1.0997180938720703, -0.7472116351127625]],
                            [[-0.8416294455528259, 0.2393561452627182]],
                            [[0.5353959202766418, -0.6356369256973267]],
                            [[0.011429687030613422, -1.0444294214248657]],
                            [[0.8294717073440552, -1.1644237041473389]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[-0.5676082372665405, 1.6229828596115112]],
                            [[-1.7003906965255737, -1.0814650058746338]],
                            [[0.5154496431350708, -0.6644119024276733]],
                            [[0.4813891053199768, -1.372441291809082]],
                            [[-1.166320562362671, -0.726149320602417]],
                            [[-0.06755267083644867, 1.1745551824569702]],
                            [[-0.39131826162338257, -0.7624845504760742]],
                            [[0.9889782667160034, -1.0488485097885132]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[0.09571779519319534, 0.8825100064277649]],
                            [[-0.209646537899971, -0.11976470053195953]],
                            [[1.8110568523406982, 0.26369625329971313]],
                            [[0.2035822570323944, -0.5185909867286682]],
                            [[1.2006571292877197, 0.2735150456428528]],
                            [[-1.4237172603607178, 0.4937574565410614]],
                            [[1.2166802883148193, 0.6768909692764282]],
                            [[-0.5895527601242065, 1.2136154174804688]],
                            [[-1.7983052730560303, -1.4686884880065918]],
                        ],
                        [
                            [[-0.7603991031646729, 0.6917804479598999]],
                            [[-1.3157434463500977, -1.3364534378051758]],
                            [[0.6037954688072205, -0.4373283088207245]],
                            [[-1.0926811695098877, 0.4675874412059784]],
                            [[-1.2689539194107056, -0.5594342947006226]],
                            [[-0.5327851176261902, -0.8017807602882385]],
                            [[0.8136177659034729, -0.8163193464279175]],
                            [[1.0433070659637451, 0.6938974857330322]],
                            [[-0.48861029744148254, 0.7178099155426025]],
                            [[-1.5572731494903564, 0.30047574639320374]],
                        ],
                        [
                            [[1.3027359247207642, 0.5769704580307007]],
                            [[-0.30500108003616333, 0.6116410493850708]],
                            [[0.993407666683197, 0.5634545683860779]],
                            [[0.5683414936065674, -0.038891635835170746]],
                            [[0.18263010680675507, -0.9309208393096924]],
                            [[-0.30102670192718506, -0.4619215428829193]],
                            [[-0.22547315061092377, 0.6620444655418396]],
                            [[0.18600444495677948, 0.3983207643032074]],
                            [[-1.3858344554901123, 1.4597420692443848]],
                            [[-0.4150679409503937, -0.09204770624637604]],
                        ],
                        [
                            [[-0.21830973029136658, 0.4716184139251709]],
                            [[-1.6000971794128418, 0.016703147441148758]],
                            [[1.486158847808838, 0.38790279626846313]],
                            [[1.3315898180007935, -0.04911533743143082]],
                            [[0.8701032996177673, -0.11769035458564758]],
                            [[0.8997407555580139, -0.6535756587982178]],
                            [[0.2050367146730423, 0.11578050255775452]],
                            [[-0.6097397208213806, -0.32293230295181274]],
                            [[0.36548155546188354, -0.3372085690498352]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[0.1818438619375229, -0.4313610792160034]],
                            [[0.22320835292339325, 0.021276840940117836]],
                            [[-1.1691735982894897, 0.35153236985206604]],
                            [[0.2640562653541565, -0.17022287845611572]],
                            [[-1.3610483407974243, -0.2239222377538681]],
                            [[-0.516293466091156, -0.22145582735538483]],
                            [[0.03286828100681305, -0.16413533687591553]],
                            [[-0.5490068793296814, 0.349784791469574]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[1.0159858465194702, -0.5081911683082581]],
                            [[0.8340901136398315, -0.21344462037086487]],
                            [[-1.8715938329696655, -0.501074492931366]],
                            [[-1.5713595151901245, -0.3148786127567291]],
                            [[0.4968576431274414, -0.6774446368217468]],
                            [[0.44643816351890564, 1.2766880989074707]],
                            [[1.2446485757827759, -0.9975428581237793]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[0.5764395594596863, -1.1536171436309814]],
                            [[1.8420573472976685, -0.009640185162425041]],
                            [[-0.07150302827358246, -0.016818424686789513]],
                            [[1.0523600578308105, 0.0008560324786230922]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                        ],
                    ]
                ),
            },
            {
                "filter_size": [],
                "n_vars": 78,
                "kernel": np.array(
                    [
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            -1.4257220029830933,
                            0.4779662489891052,
                            -0.019794760271906853,
                            -0.4925185739994049,
                            0.7232375741004944,
                            -0.03933761268854141,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.9184949398040771,
                            -0.6554914116859436,
                            -1.0997180938720703,
                            -0.8416294455528259,
                            0.5353959202766418,
                            0.011429687030613422,
                            0.8294717073440552,
                        ],
                        [
                            0.0,
                            0.0,
                            -0.5676082372665405,
                            -1.7003906965255737,
                            0.5154496431350708,
                            0.4813891053199768,
                            -1.166320562362671,
                            -0.06755267083644867,
                            -0.39131826162338257,
                            0.9889782667160034,
                        ],
                        [
                            0.0,
                            0.09571779519319534,
                            -0.209646537899971,
                            1.8110568523406982,
                            0.2035822570323944,
                            1.2006571292877197,
                            -1.4237172603607178,
                            1.2166802883148193,
                            -0.5895527601242065,
                            -1.7983052730560303,
                        ],
                        [
                            -0.7603991031646729,
                            -1.3157434463500977,
                            0.6037954688072205,
                            -1.0926811695098877,
                            -1.2689539194107056,
                            -0.5327851176261902,
                            0.8136177659034729,
                            1.0433070659637451,
                            -0.48861029744148254,
                            -1.5572731494903564,
                        ],
                        [
                            1.3027359247207642,
                            -0.30500108003616333,
                            0.993407666683197,
                            0.5683414936065674,
                            0.18263010680675507,
                            -0.30102670192718506,
                            -0.22547315061092377,
                            0.18600444495677948,
                            -1.3858344554901123,
                            -0.4150679409503937,
                        ],
                        [
                            -0.21830973029136658,
                            -1.6000971794128418,
                            1.486158847808838,
                            1.3315898180007935,
                            0.8701032996177673,
                            0.8997407555580139,
                            0.2050367146730423,
                            -0.6097397208213806,
                            0.36548155546188354,
                            0.0,
                        ],
                        [
                            0.1818438619375229,
                            0.22320835292339325,
                            -1.1691735982894897,
                            0.2640562653541565,
                            -1.3610483407974243,
                            -0.516293466091156,
                            0.03286828100681305,
                            -0.5490068793296814,
                            0.0,
                            0.0,
                        ],
                        [
                            1.0159858465194702,
                            0.8340901136398315,
                            -1.8715938329696655,
                            -1.5713595151901245,
                            0.4968576431274414,
                            0.44643816351890564,
                            1.2446485757827759,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.5764395594596863,
                            1.8420573472976685,
                            -0.07150302827358246,
                            1.0523600578308105,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                    ]
                ),
            },
        ]

        for test in test_cases:
            tf.random.set_seed(42)
            kernel_obj = icecube.IceCubeKernel(
                shape=test["filter_size"],
                get_ones=False,
                float_precision="float32",
                seed=42,
                name="HexKernel",
            )
            kernel = kernel_obj()
            var_list = kernel_obj.var_list

            self.assertEqual(len(var_list), test["n_vars"])
            self.assertTrue(np.allclose(test["kernel"], kernel, atol=1e-6))

    def test_rotated_hex_kernel(self):
        """Test RotatedHexKernel"""

        test_cases = [
            {
                "filter_size": [2, 0, 2, 1],
                "num_rotations": 3,
                "n_vars": 8,
                "kernel": np.array(
                    [
                        [
                            [[-0.0, 0.0, -0.0], [0.0, 0.0, 0.0]],
                            [
                                [
                                    -0.6020655632019043,
                                    0.27684009075164795,
                                    0.015805857256054878,
                                ],
                                [
                                    0.6445701718330383,
                                    0.2416737973690033,
                                    -0.03702780604362488,
                                ],
                            ],
                            [
                                [
                                    -0.31330278515815735,
                                    -0.015057608485221863,
                                    0.3932696282863617,
                                ],
                                [
                                    0.3354213237762451,
                                    -0.013144878670573235,
                                    -0.9212984442710876,
                                ],
                            ],
                        ],
                        [
                            [
                                [
                                    -0.4740760326385498,
                                    -0.0075770169496536255,
                                    -0.7334061861038208,
                                ],
                                [
                                    0.5075448155403137,
                                    -0.006614527199417353,
                                    1.7181239128112793,
                                ],
                            ],
                            [
                                [
                                    0.934548556804657,
                                    -0.5457363128662109,
                                    1.1384203433990479,
                                ],
                                [
                                    -1.0005258321762085,
                                    -0.4764128029346466,
                                    -2.666935920715332,
                                ],
                            ],
                            [
                                [
                                    0.012975295074284077,
                                    0.35158050060272217,
                                    -0.5774956941604614,
                                ],
                                [
                                    -0.013891325332224369,
                                    0.3069201111793518,
                                    1.352878212928772,
                                ],
                            ],
                        ],
                        [
                            [
                                [
                                    -0.4740760326385498,
                                    -0.0075770169496536255,
                                    -0.7334061861038208,
                                ],
                                [
                                    0.5075448155403137,
                                    -0.006614527199417353,
                                    1.7181239128112793,
                                ],
                            ],
                            [
                                [
                                    0.32284170389175415,
                                    0.18295539915561676,
                                    0.031410567462444305,
                                ],
                                [
                                    -0.34563368558883667,
                                    0.15971504151821136,
                                    -0.07358439266681671,
                                ],
                            ],
                            [[-0.0, 0.0, -0.0], [0.0, 0.0, 0.0]],
                        ],
                    ]
                ),
            },
            {
                "filter_size": [3, 1, 1, 2],
                "num_rotations": 1,
                "n_vars": 26,
                "kernel": np.array(
                    [
                        [
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.2444324940443039, 0.05568281188607216]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.5779832601547241, 0.13166716694831848]],
                            [[-1.4003510475158691, -0.3190062344074249]],
                            [[-0.08110759407281876, -0.01847667247056961]],
                            [[-0.46983906626701355, -0.1070314347743988]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[2.1744582653045654, 0.49535128474235535]],
                            [[0.6188783049583435, 0.1409832388162613]],
                            [[1.102797508239746, 0.2512221932411194]],
                            [[0.5738735795021057, 0.13073095679283142]],
                            [[-0.7870204448699951, -0.17928676307201385]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[-2.04158616065979, -0.46508243680000305]],
                            [[-0.04723098501563072, -0.010759429074823856]],
                            [[-1.7118033170700073, -0.38995641469955444]],
                            [[-0.023766720667481422, -0.005414164625108242]],
                            [[-1.3203843832015991, -0.3007894456386566]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[-0.6815028786659241, -0.15524938702583313]],
                            [[0.8683603405952454, 0.19781635701656342]],
                            [[-0.5913459658622742, -0.13471123576164246]],
                            [[-1.010508418083191, -0.23019830882549286]],
                            [[1.187423825263977, 0.27050042152404785]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[-0.2517136037349701, -0.05734148249030113]],
                            [[0.9959111213684082, 0.2268729954957962]],
                            [[0.013723134994506836, 0.0031261914409697056]],
                            [[0.6428269147872925, 0.1464388370513916]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.11492425203323364, 0.02618025802075863]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                        ],
                    ]
                ),
            },
        ]

        for test in test_cases:
            tf.random.set_seed(42)
            kernel_obj = rotation.RotatedHexKernel(
                filter_size=test["filter_size"],
                num_rotations=test["num_rotations"],
                float_precision="float32",
                seed=42,
                name="RotatedHexKernel",
            )
            kernel = kernel_obj()
            var_list = kernel_obj.var_list

            self.assertEqual(len(var_list), test["n_vars"])
            self.assertTrue(np.allclose(test["kernel"], kernel, atol=1e-6))

    def test_dynamic_rotated_hex_kernel(self):
        """Test DynamicRotationHexKernel"""

        test_cases = [
            {
                "filter_size": [2, 0, 2, 1],
                "azimuth": 60.0,
                "n_vars": 2,
                "kernel": np.array(
                    [
                        [
                            [[0.0], [0.0]],
                            [[-1.8633298873901367], [1.8340147733688354]],
                            [[0.8756545186042786], [-0.9188050627708435]],
                        ],
                        [
                            [[-0.8208955526351929], [-1.7109564542770386]],
                            [[-1.4257220029830933], [-1.0143787860870361]],
                            [[0.4779662489891052], [-1.6896843910217285]],
                        ],
                        [
                            [[-0.8208955526351929], [-1.7109564542770386]],
                            [[0.4115297496318817], [-0.6118844151496887]],
                            [[0.0], [0.0]],
                        ],
                    ]
                ),
            },
            {
                "filter_size": [3, 1, 1, 2],
                "azimuth": np.array(243.0),
                "n_vars": 9,
                "kernel": np.array(
                    [
                        [
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.614751935005188, 0.4126650393009186]],
                            [[0.1084856390953064, 0.07282324880361557]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[-0.07387778908014297, 0.04715276136994362]],
                            [[-0.10614082217216492, -0.23711901903152466]],
                            [[-0.8478127717971802, 0.1880636364221573]],
                            [[-0.8326864838600159, 0.2814864218235016]],
                            [[-0.03343696892261505, 0.37572962045669556]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[-0.41864079236984253, 0.26719897985458374]],
                            [[-0.03404032811522484, 0.24718335270881653]],
                            [[0.414851576089859, -0.665774405002594]],
                            [[-0.7592743039131165, -1.6560028791427612]],
                            [[0.26231634616851807, -0.5828783512115479]],
                            [[-0.005900642368942499, 0.06630522757768631]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[-0.08913522213697433, -1.225508689880371]],
                            [[0.4978506565093994, -1.6511404514312744]],
                            [[-1.4257220029830933, -1.0143787860870361]],
                            [[-0.5993744134902954, -0.17593613266944885]],
                            [[-0.365479052066803, 0.6292445063591003]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[-0.16495771706104279, -0.11208175122737885]],
                            [[0.5046157836914062, -1.3580321073532104]],
                            [[0.7387052774429321, -0.7811640501022339]],
                            [[-1.7995491027832031, 1.7375566959381104]],
                            [[0.12104412913322449, -1.0844377279281616]],
                            [[0.7807207107543945, -1.086395502090454]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[-0.9347603917121887, -0.6351298689842224]],
                            [[0.042605914175510406, -0.19909705221652985]],
                            [[0.9114534854888916, 1.712262511253357]],
                            [[-0.6210527420043945, -0.8963721394538879]],
                            [[0.1377742439508438, -0.19171684980392456]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                        ],
                        [
                            [[0.0, 0.0]],
                            [[-0.09832371771335602, 0.10526517778635025]],
                            [[-0.5571677088737488, 0.5965026021003723]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                            [[0.0, 0.0]],
                        ],
                    ]
                ),
            },
        ]

        for test in test_cases:
            tf.random.set_seed(42)
            kernel_obj = rotation.DynamicRotationHexKernel(
                filter_size=test["filter_size"],
                float_precision="float32",
                seed=42,
                name="DynamicRotationHexKernel",
            )
            kernel = kernel_obj(azimuth=test["azimuth"])
            var_list = kernel_obj.var_list

            self.assertEqual(len(var_list), test["n_vars"])
            self.assertTrue(np.allclose(test["kernel"], kernel, atol=1e-6))
