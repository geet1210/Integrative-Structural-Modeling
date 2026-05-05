# Integrative Structural Modeling Pipeline

End-to-end pipeline for integrative structural modeling of multidomain protein complexes.

AlphaFold2 prediction → XL-MS restraints → IMP/MODELLER 
refinement → MD simulations → Allostery analysis

## Pipeline stages
1. AlphaFold2/Multimer structure prediction
2. XL-MS data processing (XlinkX, Protein Prospector)
3. Restraint-driven modeling (IMP, MODELLER)
4. All-atom MD simulations (GROMACS)
5. Conformational analysis (PCA, DCCM, CNA, MM/PBSA)

## Application
PDE6 holoenzyme — four conformational states
Visual transduction pathway | Retinal disease target

## Status
🚧 In progress — manuscripts in preparation
Code and documentation will be released upon publication
