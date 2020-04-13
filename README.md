# Test-VQE-Repository
Repository for housing and testing ADAPTVQE script
Currently enabled:
  ADAPT using ROTO 2a
  
Eventually Options will be:
1. ADAPT then ROTO: Find ansatz using normal ADAPT, then implement ROTOSOLVE at end for each operator/parameter pair already in ansatz 
    2a. ADAPT using ROTO: Use ROTOSOLVE on each operator/parameter pair as it's added and use the single parameter/operator combo maximum energy step to add new parameter/operator pair, then iterate until final energy
    2b. ADAPT using ROTO: Use ROTOSOLVE on each operator/parameter pair as it's added and use all parameters that have previously been added + new parameter to choose new operator, then iterate until final energy - may not be very efficient, will require lots of ROTOSOLVe calls (could combine with gradient?)
    2c. ADAPT using ROTO: Use ROTOSOLVE on each operator/parameter pair as it's added and use the single parameter/operator combo maximum energy step to add new parameter/operator pair, then use ROTOsolve across entire operator/parameter pair list, then iterate until final energy. - will be less efficient than 2a but potentially more efficient than 2b.
3. ADAPT mix ROTO: Use ROTOSOLVE on each operator/parameter par as it's added, then classically optimize further??
	- if 2 is done with variable shots for optimal parameter measurement and variable optimizer max calls then can find best mix
