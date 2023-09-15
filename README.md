## Purpose of Code

This repository is part of a publication. It is created to calculate currrents using the non-equilibrium Green's function (NEGF) formalism for systems with onsite Coulomb Interactions in the Hartree-Fock and Hubbard One approximation. The transport is always considered to be 2 terminal.

The isolated scattering region without interactions has bipartite lattice and is constructed with the kwant and qsymm code. Therefore be sure to download the kwant,qsymm code from: https://kwant-project.org/install and https://qsymm.readthedocs.io/en/latest/index.html. Note that it is not yet possible to combine transport and Coulomb interactions with kwant, therefore we have constructed our own code. For isolated systems such an extension has been built,see the pyqula code of Jose Lado.

In our code the scattering region is attached to leads which are modelled in the wide-band limit (WBL) or they are modelled as semi-infinite (SIF) leads. For the SIF leads we consider two scenarios. The first one is that the onsite energy of the lead $\alpha$ is equal to the chemical potential of that lead $\mu_\alpha$ and the second scenario is that the onsite energy of both leads are zero. 

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
