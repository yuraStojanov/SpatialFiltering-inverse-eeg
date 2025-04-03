"""
.. _tut-eeg-mri-coords:

===========================================================
EEG source localization given electrode locations on an MRI
===========================================================

This tutorial explains how to compute the forward operator from EEG data when
the electrodes are in MRI voxel coordinates.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import nibabel
import numpy as np
from nilearn.plotting import plot_glass_brain

import mne
from mne.channels import compute_native_head_t, read_custom_montage


##############################################################################
# Prerequisites
# -------------
# For this we will assume that you have:
#
# - raw EEG data
# - your subject's MRI reconstrcted using FreeSurfer
# - an appropriate boundary element model (BEM)
# - an appropriate source space (src)
# - your EEG electrodes in Freesurfer surface RAS coordinates, stored
#   in one of the formats :func:`mne.channels.read_custom_montage` supports
#
# Let's set the paths to these files for the ``sample`` dataset, including
# a modified ``sample`` MRI showing the electrode locations plus a ``.elc``
# file corresponding to the points in MRI coords (these were `synthesized
# <https://gist.github.com/larsoner/0ac6fad57e31cb2d9caa77350a9ff366>`__,
# and thus are stored as part of the ``misc`` dataset).

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
fname_raw = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
bem_dir = subjects_dir / "sample" / "bem"
fname_bem = bem_dir / "sample-5120-5120-5120-bem-sol.fif"
fname_src = bem_dir / "sample-oct-6-src.fif"

misc_path = mne.datasets.misc.data_path()
fname_T1_electrodes = misc_path / "sample_eeg_mri" / "T1_electrodes.mgz"
fname_mon = misc_path / "sample_eeg_mri" / "sample_mri_montage.elc"

#%%
# Visualizing the MRI
# -------------------
# Let's take our MRI-with-eeg-locations and adjust the affine to put the data
# in MNI space, and plot using :func:`nilearn.plotting.plot_glass_brain`,
# which does a maximum intensity projection (easy to see the fake electrodes).
# This plotting function requires data to be in MNI space.
# Because ``img.affine`` gives the voxel-to-world (RAS) mapping, if we apply a
# RAS-to-MNI transform to it, it becomes the voxel-to-MNI transformation we
# need. Thus we create a "new" MRI image in MNI coordinates and plot it as:

img = nibabel.load(fname_T1_electrodes)  # original subject MRI w/EEG
ras_mni_t = mne.transforms.read_ras_mni_t("sample", subjects_dir)  # from FS
mni_affine = np.dot(ras_mni_t["trans"], img.affine)  # vox->ras->MNI
img_mni = nibabel.Nifti1Image(img.dataobj, mni_affine)  # now in MNI coords!
plot_glass_brain(
    img_mni,
    cmap="hot_black_bone",
    threshold=0.0,
    black_bg=True,
    resampling_interpolation="nearest",
    colorbar=True,
)

#%%
# Getting our MRI voxel EEG locations to head (and MRI surface RAS) coords
# ------------------------------------------------------------------------
# Let's load our :class:`~mne.channels.DigMontage` using
# :func:`mne.channels.read_custom_montage`, making note of the fact that
# we stored our locations in Freesurfer surface RAS (MRI) coordinates.
#
# .. dropdown:: What if my electrodes are in MRI voxels?
#     :color: warning
#     :icon: question
#
#     If you have voxel coordinates in MRI voxels, you can transform these to
#     FreeSurfer surface RAS (called "mri" in MNE) coordinates using the
#     transformations that FreeSurfer computes during reconstruction.
#     ``nibabel`` calls this transformation the ``vox2ras_tkr`` transform
#     and operates in millimeters, so we can load it, convert it to meters,
#     and then apply it::
#
#         >>> pos_vox = ...  # loaded from a file somehow
#         >>> img = nibabel.load(fname_T1)
#         >>> vox2mri_t = img.header.get_vox2ras_tkr()  # voxel -> mri trans
#         >>> pos_mri = mne.transforms.apply_trans(vox2mri_t, pos_vox)
#         >>> pos_mri /= 1000.  # mm -> m
#
#     You can also verify that these are correct (or manually convert voxels
#     to MRI coords) by looking at the points in Freeview or tkmedit.

dig_montage = read_custom_montage(fname_mon, head_size=None, coord_frame="mri")

#%%
# We can then get our transformation from the MRI coordinate frame (where our
# points are defined) to the head coordinate frame from the object.

trans = compute_native_head_t(dig_montage)


#%%
# Let's apply this digitization to our dataset, and in the process
# automatically convert our locations to the head coordinate frame, as
# shown by :meth:`~mne.io.Raw.plot_sensors`.

raw = mne.io.read_raw_fif(fname_raw)
raw.pick(picks=["eeg", "stim"]).load_data()
raw.set_montage(dig_montage)
raw.plot_sensors(show_names=True)

#%%
# Now we can do standard sensor-space operations like make joint plots of
# evoked data.

raw.set_eeg_reference(projection=True)
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events)
noise_cov = mne.compute_covariance(epochs, tmax=0.0)
evoked = epochs["1"].average()  # trigger 1 in auditory/left
evoked.plot_joint()

#%%
# Now we can actually compute the forward:

fwd = mne.make_forward_solution(
    evoked.info, trans=trans, src=fname_src, bem=fname_bem, verbose=True,
)
mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=False)
#%%
#адреса stc.vertices[] совпадают с соотв stc.data с поправкой на hemi

src_id = 2005 + 4096#2005
G = fwd['sol']['data']
s = np.zeros([1, G.shape[1]]).T
s[src_id] = 2*pow(10,-9)#17000

d = np.matmul(G,s)
evoked.data[:,:] = d

source_std = np.ones(fwd['sol']['data'].shape[1])
#%%
inv = mne.minimum_norm.make_inverse_operator( evoked.info, fwd, noise_cov, loose=0, depth=None, source_std = source_std,)
'''
idis = np.array([1160, 1161, 1190, 1226, 1227, 1228, 1229, 1262, 1291, 1292, 1293,
       1295, 1327, 1328, 1367, 1370, 1372, 1373, 1374, 1375, 1416, 1417,
       1419, 1420, 1421, 1422, 1454, 1457, 1458, 1460, 1461, 1462, 1494,
       1497, 1500, 1502, 1503, 1533, 1535, 1537, 1539, 1540, 1541, 1542,
       1579, 1580, 1582, 1583, 1585, 1586, 1587, 1589, 1621, 1625, 1626,
       1628, 1629, 1630, 1632, 1633, 1634, 1670, 1672, 1673, 1674, 1678,
       1681, 1684, 1685, 1720, 1724, 1726, 1727, 1728, 1729, 1730, 1731,
       1765, 1766, 1768, 1771, 1772, 1808, 1809, 1813, 1814, 1816, 1817,
       1818, 1847, 1848, 1849, 1850, 1851, 1852, 1854, 1882, 1885, 1887,
       1889, 1891, 1892, 1928, 1931, 1934, 1937, 1938, 1969, 1970, 1971,
       1972, 1974, 1979, 1982, 1984, 1985, 2005, 2006, 2007, 2009, 2010,
       2011, 2012, 2045, 2047, 2048, 2050, 2051, 2053, 2054, 2055, 2056,
       2057, 2093, 2096, 2097, 2100, 2137, 2140, 2143, 2145, 2147, 2180,
       2183, 2184, 2186, 2188, 2189, 2221, 2222, 2224, 2225, 2227, 2229,
       2265, 2266, 2268, 2269, 2272, 2302, 2303, 2304, 2305, 2306, 2307,
       2336, 2337, 2379, 2380, 2383, 2384, 2411, 2412, 2413, 2415, 2417,
       2418, 2451, 2452, 2454, 2455, 2456, 2459, 2460, 2461, 2462, 2463,
       2464, 2493, 2497, 2499, 2501, 2502, 2503, 2532, 2534, 2536, 2538,
       2563, 2565, 2566, 2599, 2601, 2603, 2604, 2605, 2606, 2607, 2608,
       2632, 2634, 2635, 2639, 2640, 2667, 2669, 2671, 2672, 2673, 2705,
       2707, 2708, 2729, 2753, 2754, 2755, 2788, 2789, 2790, 2819, 2820,
       2821, 2860, 2861, 2878, 2910, 2911, 2912]) # 20 (1945+4096)
'''
idis = np.array([ 686,  687,  688,  690,  717,  746,  748,  750,  783,  784,  785,
        816,  817,  818,  819,  820,  821,  847,  849,  850,  851,  853,
        854,  857,  882,  883,  884,  885,  886,  887,  907,  908,  910,
        911,  912,  913,  914,  915,  916,  946,  947,  948,  950,  952,
        953,  954,  955,  956,  957,  982,  983,  985,  987,  988,  989,
        990,  991,  992,  993,  995,  997, 1018, 1019, 1020, 1021, 1022,
       1023, 1024, 1025, 1026, 1027, 1028, 1050, 1051, 1052, 1053, 1054,
       1055, 1056, 1076, 1077, 1079, 1080, 1081, 1082, 1083, 1084, 1086,
       1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097,
       1122, 1123, 1124, 1125, 1126, 1127, 1128, 1131, 1132, 1133, 1134,
       1135, 1136, 1139, 1140, 1158, 1160, 1161, 1163, 1164, 1165, 1166,
       1167, 1169, 1170, 1171, 1172, 1174, 1188, 1189, 1190, 1191, 1192,
       1194, 1195, 1196, 1198, 1199, 1200, 1201, 1202, 1205, 1206, 1209,
       1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236,
       1238, 1239, 1240, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266,
       1268, 1269, 1270, 1271, 1274, 1275, 1276, 1289, 1290, 1291, 1292,
       1293, 1295, 1296, 1297, 1324, 1327, 1328, 1330, 1331, 1332, 1333,
       1334, 1335, 1338, 1339, 1341, 1342, 1343, 1344, 1345, 1367, 1368,
       1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1378, 1380, 1381,
       1384, 1387, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418,
       1419, 1420, 1421, 1422, 1423, 1426, 1427, 1429, 1430, 1431, 1432,
       1433, 1434, 1435, 1454, 1456, 1457, 1458, 1459, 1460, 1461, 1462,
       1465, 1468, 1469, 1470, 1471, 1492, 1494, 1495, 1497, 1499, 1500,
       1502, 1503, 1505, 1506, 1507, 1510, 1531, 1533, 1534, 1535, 1536,
       1537, 1538, 1539, 1540, 1541, 1542, 1544, 1545, 1548, 1550, 1575,
       1576, 1577, 1578, 1579, 1580, 1582, 1583, 1585, 1586, 1587, 1589,
       1590, 1591, 1592, 1593, 1594, 1595, 1596, 1619, 1621, 1623, 1625,
       1626, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1636, 1638, 1642,
       1643, 1644, 1645, 1646, 1647, 1670, 1671, 1672, 1673, 1674, 1675,
       1676, 1677, 1678, 1680, 1681, 1684, 1685, 1686, 1687, 1688, 1689,
       1692, 1693, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1726, 1727,
       1728, 1729, 1730, 1731, 1732, 1733, 1735, 1736, 1737, 1763, 1764,
       1765, 1766, 1767, 1768, 1771, 1772, 1775, 1777, 1779, 1780, 1781,
       1782, 1806, 1808, 1809, 1811, 1812, 1813, 1814, 1816, 1817, 1818,
       1820, 1821, 1822, 1824, 1826, 1828, 1845, 1846, 1847, 1848, 1849,
       1850, 1851, 1852, 1853, 1854, 1856, 1860, 1861, 1862, 1881, 1882,
       1885, 1886, 1887, 1889, 1890, 1891, 1892, 1893, 1894, 1896, 1898,
       1899, 1901, 1927, 1928, 1930, 1931, 1933, 1934, 1937, 1938, 1943,
       1944, 1945, 1968, 1969, 1970, 1971, 1972, 1974, 1976, 1977, 1978,
       1979, 1982, 1984, 1985, 1986, 1987, 1989, 1990, 2005, 2006, 2007,
       2009, 2010, 2011, 2012, 2014, 2016, 2017, 2018, 2019, 2020, 2045,
       2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057,
       2059, 2060, 2062, 2064, 2089, 2091, 2092, 2093, 2096, 2097, 2100,
       2102, 2104, 2105, 2106, 2108, 2109, 2136, 2137, 2138, 2140, 2141,
       2143, 2145, 2147, 2149, 2150, 2151, 2153, 2154, 2176, 2177, 2178,
       2180, 2181, 2183, 2184, 2186, 2187, 2188, 2189, 2192, 2193, 2195,
       2196, 2216, 2218, 2219, 2221, 2222, 2223, 2224, 2225, 2226, 2227,
       2229, 2232, 2233, 2235, 2263, 2265, 2266, 2268, 2269, 2272, 2275,
       2276, 2277, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303,
       2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2314, 2316,
       2317, 2332, 2333, 2334, 2336, 2337, 2342, 2343, 2344, 2345, 2379,
       2380, 2383, 2384, 2385, 2387, 2409, 2411, 2412, 2413, 2414, 2415,
       2416, 2417, 2418, 2423, 2425, 2426, 2447, 2448, 2449, 2451, 2452,
       2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463,
       2464, 2466, 2467, 2468, 2493, 2494, 2495, 2496, 2497, 2499, 2501,
       2502, 2503, 2505, 2506, 2507, 2509, 2510, 2531, 2532, 2534, 2536,
       2537, 2538, 2541, 2563, 2565, 2566, 2567, 2570, 2572, 2573, 2598,
       2599, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2612,
       2614, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2638, 2639, 2640,
       2667, 2669, 2670, 2671, 2672, 2673, 2678, 2705, 2706, 2707, 2708,
       2710, 2711, 2712, 2728, 2729, 2730, 2731, 2732, 2750, 2751, 2752,
       2753, 2754, 2755, 2756, 2757, 2758, 2759, 2781, 2783, 2784, 2785,
       2787, 2788, 2789, 2790, 2793, 2797, 2798, 2799, 2818, 2819, 2820,
       2821, 2822, 2824, 2825, 2856, 2857, 2858, 2859, 2860, 2861, 2862,
       2863, 2874, 2876, 2877, 2878, 2879, 2881, 2902, 2903, 2904, 2905,
       2906, 2907, 2910, 2911, 2912, 2916, 2929, 2930, 2932, 2933, 2934,
       2935, 2936, 2938, 2939, 2940, 2964, 2966, 2967, 2968, 2969, 2970,
       2994, 2995, 2997, 2998, 2999, 3000, 3001, 3003, 3020, 3022, 3023,
       3024, 3026, 3027, 3028, 3043, 3044, 3045, 3047, 3050, 3051, 3067,
       3068, 3069, 3094, 3095, 3096, 3097, 3098, 3122, 3123, 3124, 3125,
       3127, 3128, 3145, 3146, 3148, 3152, 3153, 3155, 3169, 3170, 3171,
       3172, 3173, 3174, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3232,
       3233, 3235, 3259, 3260, 3262, 3263, 3265, 3266, 3268, 3289, 3291,
       3314, 3315, 3316, 3317, 3318, 3319, 3339, 3340, 3380, 3382, 3404,
       3405, 3407, 3408, 3446, 4076, 4092])#35

#%%
from mne.minimum_norm import make_inverse_resolution_matrix, resolution_metrics



method = "MNE"


ple_array = np.array([0])
sd_array   = np.array([0])
    
ple_outside = np.array([0])
sd_outside   = np.array([0])
stc = mne.minimum_norm.apply_inverse(evoked, inv, method=method)
stc.data = abs(stc.data)


opor = mne.vertex_to_mni(stc.vertices[1][src_id-4096], 1,subject = 'sample', subjects_dir =subjects_dir ) 
snr = 3.0

lambda2 = 1.0 / snr**2
for r in np.arange(1,35,1):
    id_array = np.array([0])
    source_std = np.ones(fwd['sol']['data'].shape[1]) * pow(10,-1)
    for i in idis:
        x, y, z = mne.vertex_to_mni(stc.vertices[1][i], 1,subject = 'sample', subjects_dir =subjects_dir )
        if (abs(x-opor[0]) < r) & (abs(y-opor[1]) < r) & (abs(z-opor[2]) < r) :
            id_array = np.append(id_array,i)
    id_array = id_array[1:] +4096
    source_std[id_array] = 1000
    inv = mne.minimum_norm.make_inverse_operator( evoked.info, fwd, noise_cov, loose=0, depth=None, source_std = source_std,)
    rm_mne = make_inverse_resolution_matrix(
        fwd, inv, method=method, lambda2=lambda2
        )
    ple_mne_psf = resolution_metrics(
        rm_mne, inv["src"], function="psf", metric="peak_err"
        )
    sd_mne_psf = resolution_metrics(
        rm_mne, inv["src"], function="psf", metric="sd_ext"
        )
    summ_ple = 0
    summ_sd = 0
    for j in np.arange(4096,8193,1):
        if  not( j in id_array) :
            summ_ple = summ_ple +ple_mne_psf.data[j][0]
            summ_sd = summ_sd +sd_mne_psf.data[j][0]
    ple_outside = np.append(ple_outside,summ_ple/(4096-id_array.shape[0]))
    sd_outside   = np.append(sd_outside,summ_sd/(4096-id_array.shape[0]))

    summ_ple = 0
    summ_sd = 0
    for idi in id_array:
        summ_ple = summ_ple + ple_mne_psf.data[idi][0]
        summ_sd = summ_sd + sd_mne_psf.data[idi][0]
    ple_array = np.append(ple_array,summ_ple/id_array.shape[0])
    sd_array = np.append(sd_array,summ_sd/id_array.shape[0])
    
    print(r)
ple_array_MNE = ple_array[1:]
sd_array_MNE = sd_array[1:]
ple_outside_MNE = ple_outside[1:]
sd_outside_MNE = sd_outside[1:]

#%%
method = "sLORETA"


ple_array = np.array([0])
sd_array   = np.array([0])
    
ple_outside = np.array([0])
sd_outside   = np.array([0])
stc = mne.minimum_norm.apply_inverse(evoked, inv, method=method)
stc.data = abs(stc.data)


opor = mne.vertex_to_mni(stc.vertices[1][src_id-4096], 1,subject = 'sample', subjects_dir =subjects_dir ) 
snr = 3.0

lambda2 = 1.0 / snr**2
for r in np.arange(1,35,1):
    id_array = np.array([0])
    source_std = np.ones(fwd['sol']['data'].shape[1]) * pow(10,-1)
    for i in idis:
        x, y, z = mne.vertex_to_mni(stc.vertices[1][i], 1,subject = 'sample', subjects_dir =subjects_dir )
        if (abs(x-opor[0]) < r) & (abs(y-opor[1]) < r) & (abs(z-opor[2]) < r) :
            id_array = np.append(id_array,i)
    id_array = id_array[1:] +4096
    source_std[id_array] = 1000
    inv = mne.minimum_norm.make_inverse_operator( evoked.info, fwd, noise_cov, loose=0, depth=None, source_std = source_std,)
    rm_mne = make_inverse_resolution_matrix(
        fwd, inv, method=method, lambda2=lambda2
        )
    ple_mne_psf = resolution_metrics(
        rm_mne, inv["src"], function="psf", metric="peak_err"
        )
    sd_mne_psf = resolution_metrics(
        rm_mne, inv["src"], function="psf", metric="sd_ext"
        )
    summ_ple = 0
    summ_sd = 0
    for j in np.arange(4096,8193,1):
        if  not( j in id_array) :
            summ_ple = summ_ple +ple_mne_psf.data[j][0]
            summ_sd = summ_sd +sd_mne_psf.data[j][0]
    ple_outside = np.append(ple_outside,summ_ple/(4096-id_array.shape[0]))
    sd_outside   = np.append(sd_outside,summ_sd/(4096-id_array.shape[0]))
    summ_ple = 0
    summ_sd = 0
    for idi in id_array:
        summ_ple = summ_ple + ple_mne_psf.data[idi][0]
        summ_sd = summ_sd + sd_mne_psf.data[idi][0]
    ple_array = np.append(ple_array,summ_ple/id_array.shape[0])
    sd_array = np.append(sd_array,summ_sd/id_array.shape[0])
    
    print(r)
ple_array_sLORETA = ple_array[1:]
sd_array_sLORETA = sd_array[1:]
ple_outside_sLORETA = ple_outside[1:]
sd_outside_sLORETA = sd_outside[1:]


#%%
method = "dSPM"

ple_array = np.array([0])
sd_array   = np.array([0])
    
ple_outside = np.array([0])
sd_outside   = np.array([0])
stc = mne.minimum_norm.apply_inverse(evoked, inv, method=method)
stc.data = abs(stc.data)


opor = mne.vertex_to_mni(stc.vertices[1][src_id-4096], 1,subject = 'sample', subjects_dir =subjects_dir ) 
snr = 3.0

lambda2 = 1.0 / snr**2
for r in np.arange(1,35,1):
    id_array = np.array([0])
    source_std = np.ones(fwd['sol']['data'].shape[1]) * pow(10,-1)
    for i in idis:
        x, y, z = mne.vertex_to_mni(stc.vertices[1][i], 1,subject = 'sample', subjects_dir =subjects_dir )
        if (abs(x-opor[0]) < r) & (abs(y-opor[1]) < r) & (abs(z-opor[2]) < r) :
            id_array = np.append(id_array,i)
    id_array = id_array[1:] +4096
    source_std[id_array] = 1000
    inv = mne.minimum_norm.make_inverse_operator( evoked.info, fwd, noise_cov, loose=0, depth=None, source_std = source_std,)
    rm_mne = make_inverse_resolution_matrix(
        fwd, inv, method=method, lambda2=lambda2
        )
    ple_mne_psf = resolution_metrics(
        rm_mne, inv["src"], function="psf", metric="peak_err"
        )
    sd_mne_psf = resolution_metrics(
        rm_mne, inv["src"], function="psf", metric="sd_ext"
        )
    summ_ple = 0
    summ_sd = 0
    for j in np.arange(4096,8193,1):
        if  not( j in id_array) :
            summ_ple = summ_ple +ple_mne_psf.data[j][0]
            summ_sd = summ_sd +sd_mne_psf.data[j][0]
    ple_outside = np.append(ple_outside,summ_ple/(4096-id_array.shape[0]))
    sd_outside   = np.append(sd_outside,summ_sd/(4096-id_array.shape[0]))
    summ_ple = 0
    summ_sd = 0
    for idi in id_array:
        summ_ple = summ_ple + ple_mne_psf.data[idi][0]
        summ_sd = summ_sd + sd_mne_psf.data[idi][0]
    ple_array = np.append(ple_array,summ_ple/id_array.shape[0])
    sd_array = np.append(sd_array,summ_sd/id_array.shape[0])
    
    print(r)
ple_array_dSPM = ple_array[1:]
sd_array_dSPM = sd_array[1:]
ple_outside_dSPM = ple_outside[1:]
sd_outside_dSPM = sd_outside[1:]



#%%

inv0 = mne.minimum_norm.make_inverse_operator( evoked.info, fwd, noise_cov, loose=0, depth=None, source_std = np.ones(fwd['sol']['data'].shape[1]))

rm_mne0 = make_inverse_resolution_matrix(
fwd, inv0, method='MNE', lambda2=lambda2
)
ple_mne_psf0 = resolution_metrics(
rm_mne0, inv0["src"], function="psf", metric="peak_err"
)
sd_mne_psf0 = resolution_metrics(
rm_mne0, inv0["src"], function="psf", metric="sd_ext"
)
ple0_MNE = ple_mne_psf0.data[4097:].mean()
sd0_MNE = sd_mne_psf0.data[4097:].mean()
#%%
inv0 = mne.minimum_norm.make_inverse_operator( evoked.info, fwd, noise_cov, loose=0, depth=None, source_std = np.ones(fwd['sol']['data'].shape[1]))

rm_mne0 = make_inverse_resolution_matrix(
fwd, inv0, method='sLORETA', lambda2=lambda2
)
ple_mne_psf0 = resolution_metrics(
rm_mne0, inv0["src"], function="psf", metric="peak_err"
)
sd_mne_psf0 = resolution_metrics(
rm_mne0, inv0["src"], function="psf", metric="sd_ext"
)
ple0_sLORETA = ple_mne_psf0.data[4097:].mean()
sd0_sLORETA = sd_mne_psf0.data[4097:].mean()
#%%
inv0 = mne.minimum_norm.make_inverse_operator( evoked.info, fwd, noise_cov, loose=0, depth=None, source_std = np.ones(fwd['sol']['data'].shape[1]))

rm_mne0 = make_inverse_resolution_matrix(
fwd, inv0, method='dSPM', lambda2=lambda2
)
ple_mne_psf0 = resolution_metrics(
rm_mne0, inv0["src"], function="psf", metric="peak_err"
)
sd_mne_psf0 = resolution_metrics(
rm_mne0, inv0["src"], function="psf", metric="sd_ext"
)
ple0_dSPM = ple_mne_psf0.data[4097:].mean()
sd0_dSPM = sd_mne_psf0.data[4097:].mean()
#%%
import matplotlib.pyplot as plt
ple_plot = plt.plot(
        np.arange(1,ple_outside_MNE.shape[0]+1),
        ple_array_MNE,
        'o--', color='blue', alpha=0.8
         )
plt.plot(
        np.arange(1,ple_outside_MNE.shape[0]+1),
        ple_outside_MNE,
        'o--', color='red', alpha=0.8
         )

plt.plot(
        np.arange(1,ple_outside_MNE.shape[0]+1),
        np.ones(ple_outside.shape[0]-1)*ple0_MNE,
        '--', color='green', alpha=0.8
         )
plt.title('MNE Погрешность локализации пика (PLE)')
plt.legend(['MNE PLE внутри ОИ', 'MNEM PLE вне ОИ','MNE PLE без выбора ОИ'])
plt.xlabel('ROI radius, мм')
plt.ylabel('PLE, мм')
plt.grid(axis='x', color='0.85')
plt.grid(axis='y', color='0.85')
plt.show()
#%%
sd_plot = plt.plot(
        np.arange(1,sd_outside_MNE.shape[0]+1),
        sd_array_MNE,
        'o--', color='blue', alpha=0.8
         )
plt.plot(
        np.arange(1,sd_outside_MNE.shape[0]+1),
        sd_outside_MNE,
        'o--', color='red', alpha=0.8
         )
plt.plot(
        np.arange(1,sd_outside_MNE.shape[0]+1),
        np.ones(sd_outside_MNE.shape[0])*sd0_MNE,
        '--', color='green', alpha=0.8
         )
plt.title('MNE Пространственная дисперсия (SD)')
plt.legend(['MNE SD внутри ОИ', 'MNE SD вне ОИ','MNE SD без выбора ОИ'])
plt.xlabel('радиус ОИ, мм')
plt.ylabel('SD, мм')
plt.grid(axis='x', color='0.85')
plt.grid(axis='y', color='0.85')
plt.show()
#%%

ple_plot_dSPM = plt.plot(
        np.arange(1,ple_outside_dSPM.shape[0]+1),
        ple_array_dSPM,
        'o--', color='blue', alpha=0.8
         )
plt.plot(
        np.arange(1,ple_outside_dSPM.shape[0]+1),
        ple_outside_dSPM,
        'o--', color='red', alpha=0.8
         )

plt.plot(
        np.arange(1,ple_outside_dSPM.shape[0]+1),
        np.ones(ple_outside.shape[0]-1)*ple0_dSPM,
        '--', color='green', alpha=0.8
         )
plt.title('dSPM Погрешность локализации пика (PLE)')
plt.legend(['dSPM PLE внутри ОИ', 'dSPM PLE вне ОИ','dSPM PLE без выбора ОИ'])
plt.xlabel('радиус ОИ, мм')
plt.ylabel('PLE, мм')
plt.grid(axis='x', color='0.85')
plt.grid(axis='y', color='0.85')
plt.show()
#%%

ple_plot_dSPM = plt.plot(
        np.arange(1,ple_outside_dSPM.shape[0]+1),
        sd_array_dSPM,
        'o--', color='blue', alpha=0.8
         )
plt.plot(
        np.arange(1,ple_outside_dSPM.shape[0]+1),
        sd_outside_dSPM,
        'o--', color='red', alpha=0.8
         )

plt.plot(
        np.arange(1,ple_outside_dSPM.shape[0]+1),
        np.ones(ple_outside.shape[0]-1)*sd0_dSPM,
        '--', color='green', alpha=0.8
         )
plt.title('dSPM Пространственная дисперсия (SD)')
plt.legend(['dSPM SD внутри ОИ', 'dSPM SD вне ОИ','dSPM SD без выбора ОИ'])
plt.xlabel('радиус ОИ, мм')
plt.ylabel('SD, мм')
plt.grid(axis='x', color='0.85')
plt.grid(axis='y', color='0.85')
plt.show()
#%%
ple_plot_sLORETA = plt.plot(
        np.arange(1,ple_outside_sLORETA.shape[0]+1),
        ple_array_sLORETA,
        'o--', color='blue', alpha=0.8
         )
plt.plot(
        np.arange(1,ple_outside_dSPM.shape[0]+1),
        ple_outside_sLORETA,
        'o--', color='red', alpha=0.8
         )

plt.plot(
        np.arange(1,ple_outside_sLORETA.shape[0]+1),
        np.ones(ple_outside.shape[0]-1)*ple0_sLORETA,
        '--', color='green', alpha=0.8
         )
plt.title('sLORETA Погрешность локализации пика (PLE)')
plt.legend(['sLORETA PLE внутри ОИ', 'sLORETA PLE вне ОИ','sLORETA PLE без выбора ОИ'])
plt.xlabel('радиус ОИ, мм')
plt.ylabel('PLE, мм')
plt.grid(axis='x', color='0.85')
plt.grid(axis='y', color='0.85')
plt.show()
#%%
sd_plot_sLORETA = plt.plot(
        np.arange(1,sd_outside_sLORETA.shape[0]+1),
        sd_array_sLORETA,
        'o--', color='blue', alpha=0.8
         )
plt.plot(
        np.arange(1,sd_outside_dSPM.shape[0]+1),
        sd_outside_sLORETA,
        'o--', color='red', alpha=0.8
         )

plt.plot(
        np.arange(1,sd_outside_sLORETA.shape[0]+1),
        np.ones(sd_outside.shape[0]-1)*sd0_sLORETA,
        '--', color='green', alpha=0.8
         )
plt.title('sLORETA Пространственная дисперсия (SD)')
plt.legend(['sLORETA SD внутри ОИ', 'sLORETA SD вне ОИ','sLORETA SD без выбора ОИ'])
plt.xlabel('радиус ОИ, мм')
plt.ylabel('SD, мм')
plt.grid(axis='x', color='0.85')
plt.grid(axis='y', color='0.85')
plt.show()
#%%
brain_ple_mne = ple_mne_psf.plot(
    "sample",
    "inflated",
    "rh",
    subjects_dir=subjects_dir,
    figure=0,
    clim=dict(kind="value", lims=(0, 2, 4)),
)
brain_ple_mne.add_text(0.1, 0.9, "PLE tweaked MNE 5", "title", font_size=16)
brain_sd_mne = sd_mne_psf.plot(
    "sample",
    "inflated",
    "rh",
    subjects_dir=subjects_dir,
    figure=1,
    clim=dict(kind="value", lims=(0, 2, 4)),
)
#%%

ple_plot = plt.plot(
        np.arange(1,ple_outside_MNE.shape[0]+1),
        ple_array_MNE,
        'o--', color='blue', alpha=0.8
         )
plt.plot(
        np.arange(1,ple_outside_MNE.shape[0]+1),
        ple_outside_MNE,
        'o--', color='red', alpha=0.8
         )

plt.plot(
        np.arange(1,ple_outside_MNE.shape[0]+1),
        np.ones(ple_outside.shape[0]-1)*ple0_MNE,
        '--', color='green', alpha=0.8
         )
plt.plot(
        np.arange(1,ple_outside_dSPM.shape[0]+1),
        np.ones(ple_outside.shape[0]-1)*ple0_dSPM,
        '--', color='cyan', alpha=0.8
         )
plt.plot(
        np.arange(1,ple_outside_sLORETA.shape[0]+1),
        np.ones(ple_outside.shape[0]-1)*ple0_sLORETA,
        '--', color='magenta', alpha=0.8
         )
plt.title('Погрешность локализации пика (PLE)')
plt.legend(['MNE PLE внутри ОИ', 'MNE PLE вне ОИ','MNE PLE без выбора ОИ', 'dSPM PLE', 'sLORETA PLE'])
plt.xlabel('радиус ОИ, мм')
plt.ylabel('PLE, мм')
plt.grid(axis='x', color='0.85')
plt.grid(axis='y', color='0.85')
plt.show()
#%%
sd_plot = plt.plot(
        np.arange(1,sd_outside_MNE.shape[0]+1),
        sd_array_MNE,
        'o--', color='blue', alpha=0.8
         )
plt.plot(
        np.arange(1,sd_outside_MNE.shape[0]+1),
        sd_outside_MNE,
        'o--', color='red', alpha=0.8
         )
plt.plot(
        np.arange(1,sd_outside_MNE.shape[0]+1),
        np.ones(sd_outside_MNE.shape[0])*sd0_MNE,
        '--', color='green', alpha=0.8
         )
plt.plot(
        np.arange(1,ple_outside_dSPM.shape[0]+1),
        np.ones(ple_outside.shape[0]-1)*sd0_dSPM,
        '--', color='cyan', alpha=0.8
         )
plt.plot(
        np.arange(1,sd_outside_sLORETA.shape[0]+1),
        np.ones(sd_outside.shape[0]-1)*sd0_sLORETA,
        '--', color='magenta', alpha=0.8
         )
plt.title('Пространственная дисперсия (SD)')
plt.legend(['MNE SD внутри ОИ', 'MNE SD вне ОИ','MNE SD без выбора ОИ','dSPM SD', 'sLORETA SD'])
plt.xlabel('радиус ОИ, мм')
plt.ylabel('SD, мм')
plt.grid(axis='x', color='0.85')
plt.grid(axis='y', color='0.85')
plt.show()