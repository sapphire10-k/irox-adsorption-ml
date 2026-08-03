### Modelling feature subset

The full dictionary above records all descriptors computed. 4 of them are used for modelling, chosen on physical grounds **before** any performance comparison and not revised in response to results (data-driven selection at this sample size would leak).

| feature | why this one |
| --- | --- |
| `gcn_mean` | Generalised coordination number of the binding oxygen. Coordination-based descriptors are the standard proxy for adsorption strength on oxide surfaces: under-coordinated sites bind more strongly. |
| `nn1_dist_A` | Nearest-neighbour distance from the binding oxygen. Captures the local bond-length environment independently of the coordination count. |
| `n_bridge_sites` | Site type: how many binding oxygens are bridging (Ir coordination 2) rather than three-fold or terminal. Bridging versus in-plane oxygen is the primary site distinction on rutile (110). |
| `n_adsorbates` | Coverage, 1H versus 2H. Two adsorbates interact laterally, which is a different energetic regime from an isolated adsorbate. Constant across the present all-1H training set and kept deliberately, so that it is already in place when a pair converges. |
