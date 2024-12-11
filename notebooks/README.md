# Running the Notebooks

> Read carefully the instructions below before attempting to run the notebooks!

Since `cimr-rgb` and its internal routines rely on the configuration file and a
set of data (L1B, antenna patterns etc.), the user needs to first specify where
this data is located and the underlying folders should not be empty. Otherwise,
the code will crush.

In order to make the results reproducible, the user will have to create a
`cimr_rgb` directory inside `.local` of the current user. The data should be
placed in there.

For example, if the current user has the name `bob`, then the following
directories should be created and populated with values:

```
/home/bob/.local/cimr_rgb
```

and inside of `cimr_rgb`

```
 dpr/L1B/SMAP/SMAP_L1B_TB_47224_D_20231204T132117_R18290_001.h5
 dpr/L1B/SMAP/SMAP_L1B_TB_47185_D_20231201T212120_R18290_001.h5
 dpr/L1B/SMAP/SMAP_L1B_TB_47172_D_20231201T000114_R18290_001.h5
 dpr/L1B/CIMR/SCEPS_l1b_sceps_geo_polar_scene_1_unfiltered_tot_minimal_nom_nedt_apc_tot_v2p1.nc
 dpr/L1B/CIMR/SCEPS_l1b_sceps_geo_central_america_scene_1_unfiltered_tot_minimal_nom_nedt_apc_tot_v2p1.nc
 dpr/antenna_patterns/SMAP/RadiometerAntPattern_170830_v011.h5

dpr/antenna_patterns/CIMR/X:
 CIMR-PAP-FR-X3-TPv0.3.h5   CIMR-PAP-FR-X1-TPv0.3.h5
 CIMR-PAP-FR-X2-TPv0.3.h5   CIMR-PAP-FR-X0-TPv0.3.h5

dpr/antenna_patterns/CIMR/L:
 CIMR-PAP-FR-L0-TPv0.3.h5

dpr/antenna_patterns/CIMR/KU:
 CIMR-PAP-FR-K7-TPv0.3.h5   CIMR-PAP-FR-K4-TPv0.3.h5   CIMR-PAP-FR-K1-TPv0.3.h5
 CIMR-PAP-FR-K6-TPv0.3.h5   CIMR-PAP-FR-K3-TPv0.3.h5   CIMR-PAP-FR-K0-TPv0.3.h5
 CIMR-PAP-FR-K5-TPv0.3.h5   CIMR-PAP-FR-K2-TPv0.3.h5

dpr/antenna_patterns/CIMR/KA:
 CIMR-PAP-FR-KA7-TPv0.3.h5   CIMR-PAP-FR-KA4-TPv0.3.h5   CIMR-PAP-FR-KA1-TPv0.3.h5
 CIMR-PAP-FR-KA6-TPv0.3.h5   CIMR-PAP-FR-KA3-TPv0.3.h5   CIMR-PAP-FR-KA0-TPv0.3.h5
 CIMR-PAP-FR-KA5-TPv0.3.h5   CIMR-PAP-FR-KA2-TPv0.3.h5

dpr/antenna_patterns/CIMR/C:
 CIMR-PAP-FR-C3-TPv0.3.h5   CIMR-PAP-FR-C1-TPv0.3.h5
 CIMR-PAP-FR-C2-TPv0.3.h5   CIMR-PAP-FR-C0-TPv0.3.h5
```

Once this is done, the user can run the notebooks, whose output will be located
inside the current notebooks' folder.
