# This file has been generated automatically by generate_slakotables.py
# from /local_1tb/home/humeniuka/DFTB-0.1.0/DFTB/SlaterKoster/confined_pseudo_atoms/h.pyc and /local_1tb/home/humeniuka/DFTB-0.1.0/DFTB/SlaterKoster/confined_pseudo_atoms/li.pyc.
from numpy import array
Z1 = 1
Z2 = 3
# overlaps S[(l1,l2,i)] and hamilton matrix elements H[(l1,l2,i)]
# l1 and l2 are the angular quantum numbers of valence orbitals
# on atom1 and atom2 respectively.
# i enumerates the Slater-Koster integrals:
index2symbol = \
{0: 'ss\\sigma'}
# grid for distance d between atomic centers
d = \
array([  0.        ,   0.04016064,   0.08032129,   0.12048193,
         0.16064257,   0.20080321,   0.24096386,   0.2811245 ,
         0.32128514,   0.36144578,   0.40160643,   0.44176707,
         0.48192771,   0.52208835,   0.562249  ,   0.60240964,
         0.64257028,   0.68273092,   0.72289157,   0.76305221,
         0.80321285,   0.84337349,   0.88353414,   0.92369478,
         0.96385542,   1.00401606,   1.04417671,   1.08433735,
         1.12449799,   1.16465863,   1.20481928,   1.24497992,
         1.28514056,   1.3253012 ,   1.36546185,   1.40562249,
         1.44578313,   1.48594378,   1.52610442,   1.56626506,
         1.6064257 ,   1.64658635,   1.68674699,   1.72690763,
         1.76706827,   1.80722892,   1.84738956,   1.8875502 ,
         1.92771084,   1.96787149,   2.00803213,   2.04819277,
         2.08835341,   2.12851406,   2.1686747 ,   2.20883534,
         2.24899598,   2.28915663,   2.32931727,   2.36947791,
         2.40963855,   2.4497992 ,   2.48995984,   2.53012048,
         2.57028112,   2.61044177,   2.65060241,   2.69076305,
         2.73092369,   2.77108434,   2.81124498,   2.85140562,
         2.89156627,   2.93172691,   2.97188755,   3.01204819,
         3.05220884,   3.09236948,   3.13253012,   3.17269076,
         3.21285141,   3.25301205,   3.29317269,   3.33333333,
         3.37349398,   3.41365462,   3.45381526,   3.4939759 ,
         3.53413655,   3.57429719,   3.61445783,   3.65461847,
         3.69477912,   3.73493976,   3.7751004 ,   3.81526104,
         3.85542169,   3.89558233,   3.93574297,   3.97590361,
         4.01606426,   4.0562249 ,   4.09638554,   4.13654618,
         4.17670683,   4.21686747,   4.25702811,   4.29718876,
         4.3373494 ,   4.37751004,   4.41767068,   4.45783133,
         4.49799197,   4.53815261,   4.57831325,   4.6184739 ,
         4.65863454,   4.69879518,   4.73895582,   4.77911647,
         4.81927711,   4.85943775,   4.89959839,   4.93975904,
         4.97991968,   5.02008032,   5.06024096,   5.10040161,
         5.14056225,   5.18072289,   5.22088353,   5.26104418,
         5.30120482,   5.34136546,   5.3815261 ,   5.42168675,
         5.46184739,   5.50200803,   5.54216867,   5.58232932,
         5.62248996,   5.6626506 ,   5.70281124,   5.74297189,
         5.78313253,   5.82329317,   5.86345382,   5.90361446,
         5.9437751 ,   5.98393574,   6.02409639,   6.06425703,
         6.10441767,   6.14457831,   6.18473896,   6.2248996 ,
         6.26506024,   6.30522088,   6.34538153,   6.38554217,
         6.42570281,   6.46586345,   6.5060241 ,   6.54618474,
         6.58634538,   6.62650602,   6.66666667,   6.70682731,
         6.74698795,   6.78714859,   6.82730924,   6.86746988,
         6.90763052,   6.94779116,   6.98795181,   7.02811245,
         7.06827309,   7.10843373,   7.14859438,   7.18875502,
         7.22891566,   7.26907631,   7.30923695,   7.34939759,
         7.38955823,   7.42971888,   7.46987952,   7.51004016,
         7.5502008 ,   7.59036145,   7.63052209,   7.67068273,
         7.71084337,   7.75100402,   7.79116466,   7.8313253 ,
         7.87148594,   7.91164659,   7.95180723,   7.99196787,
         8.03212851,   8.07228916,   8.1124498 ,   8.15261044,
         8.19277108,   8.23293173,   8.27309237,   8.31325301,
         8.35341365,   8.3935743 ,   8.43373494,   8.47389558,
         8.51405622,   8.55421687,   8.59437751,   8.63453815,
         8.6746988 ,   8.71485944,   8.75502008,   8.79518072,
         8.83534137,   8.87550201,   8.91566265,   8.95582329,
         8.99598394,   9.03614458,   9.07630522,   9.11646586,
         9.15662651,   9.19678715,   9.23694779,   9.27710843,
         9.31726908,   9.35742972,   9.39759036,   9.437751  ,
         9.47791165,   9.51807229,   9.55823293,   9.59839357,
         9.63855422,   9.67871486,   9.7188755 ,   9.75903614,
         9.79919679,   9.83935743,   9.87951807,   9.91967871,
         9.95983936,  10.        ])
# overlaps
S = \
{(0, 0, 0): array([  2.45214398e-01,   2.45459062e-01,   2.46168671e-01,
         2.47341182e-01,   2.48955119e-01,   2.50996070e-01,
         2.53435849e-01,   2.56253879e-01,   2.59417571e-01,
         2.62897883e-01,   2.66666873e-01,   2.70691707e-01,
         2.74940728e-01,   2.79382093e-01,   2.83988843e-01,
         2.88725094e-01,   2.93560942e-01,   2.98472849e-01,
         3.03427652e-01,   3.08399776e-01,   3.13364689e-01,
         3.18294171e-01,   3.23168911e-01,   3.27963876e-01,
         3.32659945e-01,   3.37236666e-01,   3.41673287e-01,
         3.45956726e-01,   3.50068975e-01,   3.53997292e-01,
         3.57727090e-01,   3.61247243e-01,   3.64546794e-01,
         3.67615602e-01,   3.70448231e-01,   3.73034950e-01,
         3.75370971e-01,   3.77450666e-01,   3.79270689e-01,
         3.80827695e-01,   3.82119803e-01,   3.83145888e-01,
         3.83906384e-01,   3.84400786e-01,   3.84630676e-01,
         3.84597706e-01,   3.84306467e-01,   3.83757418e-01,
         3.82956544e-01,   3.81908352e-01,   3.80616765e-01,
         3.79087928e-01,   3.77328019e-01,   3.75341987e-01,
         3.73137912e-01,   3.70721462e-01,   3.68102154e-01,
         3.65285549e-01,   3.62279384e-01,   3.59092004e-01,
         3.55731469e-01,   3.52205758e-01,   3.48523341e-01,
         3.44692386e-01,   3.40721533e-01,   3.36617934e-01,
         3.32391336e-01,   3.28048952e-01,   3.23599964e-01,
         3.19052214e-01,   3.14412998e-01,   3.09690703e-01,
         3.04893620e-01,   3.00028613e-01,   2.95104457e-01,
         2.90126722e-01,   2.85103622e-01,   2.80042506e-01,
         2.74948934e-01,   2.69830370e-01,   2.64692464e-01,
         2.59542587e-01,   2.54385233e-01,   2.49226793e-01,
         2.44072425e-01,   2.38928275e-01,   2.33798262e-01,
         2.28687748e-01,   2.23600955e-01,   2.18542612e-01,
         2.13516434e-01,   2.08526971e-01,   2.03577412e-01,
         1.98671239e-01,   1.93811803e-01,   1.89002301e-01,
         1.84245496e-01,   1.79543857e-01,   1.74900415e-01,
         1.70317025e-01,   1.65796009e-01,   1.61338694e-01,
         1.56947605e-01,   1.52623953e-01,   1.48368950e-01,
         1.44184362e-01,   1.40071017e-01,   1.36030022e-01,
         1.32062157e-01,   1.28168129e-01,   1.24348622e-01,
         1.20604103e-01,   1.16934802e-01,   1.13341042e-01,
         1.09822888e-01,   1.06380566e-01,   1.03013758e-01,
         9.97223780e-02,   9.65061991e-02,   9.33649808e-02,
         9.02982305e-02,   8.73055526e-02,   8.43865022e-02,
         8.15401948e-02,   7.87661081e-02,   7.60635464e-02,
         7.34317058e-02,   7.08698217e-02,   6.83769746e-02,
         6.59522788e-02,   6.35948344e-02,   6.13036312e-02,
         5.90777411e-02,   5.69159581e-02,   5.48173345e-02,
         5.27807800e-02,   5.08052202e-02,   4.88894871e-02,
         4.70324717e-02,   4.52330052e-02,   4.34899572e-02,
         4.18021694e-02,   4.01684589e-02,   3.85876492e-02,
         3.70585862e-02,   3.55800659e-02,   3.41509124e-02,
         3.27699726e-02,   3.14360910e-02,   3.01480611e-02,
         2.89047439e-02,   2.77049986e-02,   2.65476846e-02,
         2.54316724e-02,   2.43558617e-02,   2.33191181e-02,
         2.23203677e-02,   2.13585325e-02,   2.04325446e-02,
         1.95413597e-02,   1.86839465e-02,   1.78592978e-02,
         1.70664088e-02,   1.63042975e-02,   1.55720060e-02,
         1.48685882e-02,   1.41931171e-02,   1.35446857e-02,
         1.29224080e-02,   1.23254155e-02,   1.17528608e-02,
         1.12039061e-02,   1.06777465e-02,   1.01735870e-02,
         9.69065674e-03,   9.22820209e-03,   8.78548911e-03,
         8.36180399e-03,   7.95645103e-03,   7.56875416e-03,
         7.19806023e-03,   6.84371829e-03,   6.50511386e-03,
         6.18164528e-03,   5.87272409e-03,   5.57778395e-03,
         5.29627575e-03,   5.02766648e-03,   4.77144012e-03,
         4.52709693e-03,   4.29415361e-03,   4.07214210e-03,
         3.86061099e-03,   3.65912359e-03,   3.46725797e-03,
         3.28460725e-03,   3.11077842e-03,   2.94539304e-03,
         2.78808504e-03,   2.63850370e-03,   2.49630787e-03,
         2.36117193e-03,   2.23278156e-03,   2.11083445e-03,
         1.99503943e-03,   1.88511696e-03,   1.78079853e-03,
         1.68182592e-03,   1.58795126e-03,   1.49893666e-03,
         1.41455391e-03,   1.33458406e-03,   1.25881719e-03,
         1.18705218e-03,   1.11909632e-03,   1.05476502e-03,
         9.93881648e-04,   9.36277131e-04,   8.81789757e-04,
         8.30264877e-04,   7.81554613e-04,   7.35517707e-04,
         6.92019222e-04,   6.50930308e-04,   6.12127944e-04,
         5.75494768e-04,   5.40918808e-04,   5.08293298e-04,
         4.77516452e-04,   4.48491309e-04,   4.21125490e-04,
         3.95331051e-04,   3.71024178e-04,   3.48125222e-04,
         3.26558364e-04,   3.06251505e-04,   2.87136103e-04,
         2.69147028e-04,   2.52222391e-04,   2.36303415e-04,
         2.21334291e-04,   2.07262067e-04,   1.94036482e-04,
         1.81609871e-04,   1.69937039e-04,   1.58975139e-04,
         1.48683575e-04,   1.39023891e-04,   1.29959659e-04,
         1.21456403e-04])}
# hamiltonian matrix elements
H = \
{(0, 0, 0): array([  1.83016006e-01,   1.80949951e-01,   1.76224258e-01,
         1.68994097e-01,   1.59776377e-01,   1.48963808e-01,
         1.36942600e-01,   1.23994952e-01,   1.10393746e-01,
         9.63513697e-02,   8.20498985e-02,   6.76486179e-02,
         5.32736760e-02,   3.90344071e-02,   2.50147304e-02,
         1.12937155e-02,  -2.07029244e-03,  -1.50340880e-02,
        -2.75575093e-02,  -3.96119557e-02,  -5.11799300e-02,
        -6.22421631e-02,  -7.27926630e-02,  -8.28257962e-02,
        -9.23410803e-02,  -1.01340902e-01,  -1.09828878e-01,
        -1.17815789e-01,  -1.25309077e-01,  -1.32320870e-01,
        -1.38862259e-01,  -1.44948105e-01,  -1.50591592e-01,
        -1.55806666e-01,  -1.60610621e-01,  -1.65016666e-01,
        -1.69040964e-01,  -1.72698280e-01,  -1.76003402e-01,
        -1.78971269e-01,  -1.81616186e-01,  -1.83952304e-01,
        -1.85992949e-01,  -1.87751662e-01,  -1.89240576e-01,
        -1.90472014e-01,  -1.91459832e-01,  -1.92213287e-01,
        -1.92745216e-01,  -1.93066917e-01,  -1.93187068e-01,
        -1.93117394e-01,  -1.92867802e-01,  -1.92446297e-01,
        -1.91863537e-01,  -1.91126939e-01,  -1.90247295e-01,
        -1.89231252e-01,  -1.88086705e-01,  -1.86821691e-01,
        -1.85443972e-01,  -1.83960151e-01,  -1.82377877e-01,
        -1.80703586e-01,  -1.78943596e-01,  -1.77103715e-01,
        -1.75190808e-01,  -1.73210006e-01,  -1.71167326e-01,
        -1.69067866e-01,  -1.66916394e-01,  -1.64717616e-01,
        -1.62477160e-01,  -1.60198528e-01,  -1.57887187e-01,
        -1.55545548e-01,  -1.53178402e-01,  -1.50790022e-01,
        -1.48382628e-01,  -1.45960217e-01,  -1.43525519e-01,
        -1.41082618e-01,  -1.38632800e-01,  -1.36179701e-01,
        -1.33725411e-01,  -1.31273234e-01,  -1.28824072e-01,
        -1.26380957e-01,  -1.23945170e-01,  -1.21519192e-01,
        -1.19104277e-01,  -1.16702913e-01,  -1.14315826e-01,
        -1.11944638e-01,  -1.09590735e-01,  -1.07255538e-01,
        -1.04940212e-01,  -1.02645442e-01,  -1.00373125e-01,
        -9.81234925e-02,  -9.58978951e-02,  -9.36961966e-02,
        -9.15203788e-02,  -8.93705688e-02,  -8.72468954e-02,
        -8.51509292e-02,  -8.30825365e-02,  -8.10425577e-02,
        -7.90312959e-02,  -7.70491825e-02,  -7.50966430e-02,
        -7.31741429e-02,  -7.12816683e-02,  -6.94196933e-02,
        -6.75882826e-02,  -6.57879758e-02,  -6.40184087e-02,
        -6.22798501e-02,  -6.05724224e-02,  -5.88963508e-02,
        -5.72514457e-02,  -5.56378916e-02,  -5.40558074e-02,
        -5.25046530e-02,  -5.09845627e-02,  -4.94955338e-02,
        -4.80373042e-02,  -4.66099257e-02,  -4.52130319e-02,
        -4.38465281e-02,  -4.25102851e-02,  -4.12039766e-02,
        -3.99275410e-02,  -3.86803652e-02,  -3.74624188e-02,
        -3.62733826e-02,  -3.51130231e-02,  -3.39809189e-02,
        -3.28767822e-02,  -3.18002263e-02,  -3.07509708e-02,
        -2.97286605e-02,  -2.87328999e-02,  -2.77632978e-02,
        -2.68195411e-02,  -2.59011499e-02,  -2.50077288e-02,
        -2.41389258e-02,  -2.32943752e-02,  -2.24735242e-02,
        -2.16760024e-02,  -2.09014111e-02,  -2.01493253e-02,
        -1.94193090e-02,  -1.87109781e-02,  -1.80238187e-02,
        -1.73574477e-02,  -1.67114354e-02,  -1.60853326e-02,
        -1.54787173e-02,  -1.48911609e-02,  -1.43222646e-02,
        -1.37715753e-02,  -1.32386636e-02,  -1.27231284e-02,
        -1.22245425e-02,  -1.17424927e-02,  -1.12765628e-02,
        -1.08263576e-02,  -1.03914760e-02,  -9.97153419e-03,
        -9.56610419e-03,  -9.17482782e-03,  -8.79730592e-03,
        -8.43316922e-03,  -8.08204168e-03,  -7.74355743e-03,
        -7.41735929e-03,  -7.10309271e-03,  -6.80041055e-03,
        -6.50898967e-03,  -6.22844826e-03,  -5.95848181e-03,
        -5.69877284e-03,  -5.44898929e-03,  -5.20882445e-03,
        -4.97797772e-03,  -4.75615119e-03,  -4.54305372e-03,
        -4.33839998e-03,  -4.14191180e-03,  -3.95331441e-03,
        -3.77234455e-03,  -3.59874329e-03,  -3.43225660e-03,
        -3.27263931e-03,  -3.11965053e-03,  -2.97305896e-03,
        -2.83263285e-03,  -2.69815773e-03,  -2.56940804e-03,
        -2.44617949e-03,  -2.32826802e-03,  -2.21547708e-03,
        -2.10761252e-03,  -2.00448882e-03,  -1.90592604e-03,
        -1.81174840e-03,  -1.72178580e-03,  -1.63587378e-03,
        -1.55385301e-03,  -1.47556894e-03,  -1.40087196e-03,
        -1.32961783e-03,  -1.26166658e-03,  -1.19688300e-03,
        -1.13513678e-03,  -1.07630172e-03,  -1.02025627e-03,
        -9.66883026e-04,  -9.16068443e-04,  -8.67703206e-04,
        -8.21682100e-04,  -7.77903674e-04,  -7.36270096e-04,
        -6.96687175e-04,  -6.59064293e-04,  -6.23314227e-04,
        -5.89353056e-04,  -5.57100184e-04,  -5.26478155e-04,
        -4.97412705e-04,  -4.69831851e-04,  -4.43667175e-04,
        -4.18852748e-04,  -3.95325335e-04,  -3.73024329e-04,
        -3.51891597e-04,  -3.31871417e-04,  -3.12910401e-04,
        -2.94957399e-04,  -2.77963519e-04,  -2.61881881e-04,
        -2.46667683e-04,  -2.32278098e-04,  -2.18672080e-04,
        -2.05810509e-04,  -1.93655987e-04,  -1.82172743e-04,
        -1.71326666e-04])}
# dipoles
Dipole = \
{(0, 0, 3): array([ -1.90263635e-18,   6.04957351e-03,   1.20559018e-02,
         1.79764363e-02,   2.37705445e-02,   2.93988263e-02,
         3.48250096e-02,   4.00143593e-02,   4.49361503e-02,
         4.95620843e-02,   5.38664837e-02,   5.78275175e-02,
         6.14262737e-02,   6.46470129e-02,   6.74757419e-02,
         6.99039775e-02,   7.19247790e-02,   7.35320012e-02,
         7.47255509e-02,   7.55056464e-02,   7.58747674e-02,
         7.58394664e-02,   7.54051801e-02,   7.45821650e-02,
         7.33802408e-02,   7.18120793e-02,   6.98925741e-02,
         6.76345448e-02,   6.50552118e-02,   6.21704942e-02,
         5.89989184e-02,   5.55583362e-02,   5.18679307e-02,
         4.79472558e-02,   4.38141470e-02,   3.94899980e-02,
         3.49936416e-02,   3.03454349e-02,   2.55644539e-02,
         2.06704190e-02,   1.56823112e-02,   1.06188671e-02,
         5.49785392e-03,   3.37874818e-04,  -4.84401819e-03,
        -1.00308231e-02,  -1.52081678e-02,  -2.03581856e-02,
        -2.54680466e-02,  -3.05239552e-02,  -3.55116554e-02,
        -4.04195904e-02,  -4.52363972e-02,  -4.99499734e-02,
        -5.45517776e-02,  -5.90313904e-02,  -6.33828182e-02,
        -6.75964877e-02,  -7.16655756e-02,  -7.55846149e-02,
        -7.93483940e-02,  -8.29522020e-02,  -8.63924330e-02,
        -8.96658679e-02,  -9.27703758e-02,  -9.57027145e-02,
        -9.84633736e-02,  -1.01050378e-01,  -1.03464827e-01,
        -1.05706729e-01,  -1.07776046e-01,  -1.09674644e-01,
        -1.11405043e-01,  -1.12968242e-01,  -1.14368615e-01,
        -1.15606465e-01,  -1.16686475e-01,  -1.17612705e-01,
        -1.18387007e-01,  -1.19014622e-01,  -1.19498565e-01,
        -1.19845032e-01,  -1.20056156e-01,  -1.20137538e-01,
        -1.20093348e-01,  -1.19929712e-01,  -1.19649591e-01,
        -1.19258703e-01,  -1.18761283e-01,  -1.18162802e-01,
        -1.17467368e-01,  -1.16680929e-01,  -1.15807324e-01,
        -1.14851342e-01,  -1.13817764e-01,  -1.12711422e-01,
        -1.11536685e-01,  -1.10297631e-01,  -1.08999649e-01,
        -1.07646200e-01,  -1.06241863e-01,  -1.04789489e-01,
        -1.03294562e-01,  -1.01760140e-01,  -1.00189495e-01,
        -9.85872340e-02,  -9.69562068e-02,  -9.53000339e-02,
        -9.36218033e-02,  -9.19246402e-02,  -9.02117280e-02,
        -8.84859483e-02,  -8.67498103e-02,  -8.50061264e-02,
        -8.32572687e-02,  -8.15060828e-02,  -7.97542858e-02,
        -7.80041821e-02,  -7.62579047e-02,  -7.45175769e-02,
        -7.27848314e-02,  -7.10615927e-02,  -6.93497418e-02,
        -6.76501418e-02,  -6.59645792e-02,  -6.42944616e-02,
        -6.26409756e-02,  -6.10054258e-02,  -5.93887676e-02,
        -5.77920471e-02,  -5.62163064e-02,  -5.46623063e-02,
        -5.31310308e-02,  -5.16227304e-02,  -5.01383463e-02,
        -4.86784232e-02,  -4.72435442e-02,  -4.58340244e-02,
        -4.44503167e-02,  -4.30926993e-02,  -4.17615188e-02,
        -4.04570217e-02,  -3.91793514e-02,  -3.79286364e-02,
        -3.67050286e-02,  -3.55084988e-02,  -3.43390573e-02,
        -3.31967286e-02,  -3.20814856e-02,  -3.09931059e-02,
        -2.99315066e-02,  -2.88965474e-02,  -2.78880357e-02,
        -2.69057514e-02,  -2.59495103e-02,  -2.50189556e-02,
        -2.41138585e-02,  -2.32339135e-02,  -2.23787856e-02,
        -2.15481456e-02,  -2.07416432e-02,  -1.99589343e-02,
        -1.91996192e-02,  -1.84633018e-02,  -1.77495967e-02,
        -1.70580895e-02,  -1.63883583e-02,  -1.57399816e-02,
        -1.51125369e-02,  -1.45055920e-02,  -1.39187173e-02,
        -1.33514479e-02,  -1.28033618e-02,  -1.22740057e-02,
        -1.17629393e-02,  -1.12697164e-02,  -1.07938939e-02,
        -1.03350328e-02,  -9.89269305e-03,  -9.46643981e-03,
        -9.05585701e-03,  -8.66047333e-03,  -8.27988574e-03,
        -7.91367829e-03,  -7.56142187e-03,  -7.22270974e-03,
        -6.89713952e-03,  -6.58431178e-03,  -6.28383363e-03,
        -5.99531793e-03,  -5.71838477e-03,  -5.45265888e-03,
        -5.19777622e-03,  -4.95337772e-03,  -4.71911081e-03,
        -4.49463243e-03,  -4.27960536e-03,  -4.07370255e-03,
        -3.87659971e-03,  -3.68798855e-03,  -3.50755524e-03,
        -3.33500555e-03,  -3.17004896e-03,  -3.01240319e-03,
        -2.86179199e-03,  -2.71794827e-03,  -2.58061271e-03,
        -2.44953257e-03,  -2.32446273e-03,  -2.20516556e-03,
        -2.09141078e-03,  -1.98297493e-03,  -1.87964166e-03,
        -1.78120185e-03,  -1.68745288e-03,  -1.59819882e-03,
        -1.51325059e-03,  -1.43242532e-03,  -1.35554662e-03,
        -1.28244420e-03,  -1.21295367e-03,  -1.14691661e-03,
        -1.08418051e-03,  -1.02459842e-03,  -9.68028781e-04,
        -9.14335408e-04,  -8.63387319e-04,  -8.15058563e-04,
        -7.69228096e-04,  -7.25779698e-04,  -6.84601793e-04,
        -6.45587384e-04,  -6.08633393e-04,  -5.73641438e-04,
        -5.40517043e-04,  -5.09169688e-04,  -4.79512700e-04,
        -4.51463095e-04,  -4.24941452e-04,  -3.99871789e-04,
        -3.76181437e-04,  -3.53800997e-04,  -3.32664096e-04,
        -3.12707368e-04,  -2.93870338e-04,  -2.76095228e-04,
        -2.59326979e-04,  -2.43513072e-04,  -2.28603410e-04,
        -2.14550283e-04])}
