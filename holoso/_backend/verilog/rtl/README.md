# Support RTL sources

This directory contains the RTL files shipped as part of a single `holoso_support.v` megafile with every
Holoso-synthesized RTL module, plus inline helpers spliced into generated modules.
The support megafile is always the same regardless of the synthesis output,
allowing designs that use multiple Holoso modules to include a single support `.v` file shared by all of them.

The subdirectories (some of them) are automatically vendored from third-party sources; see their READMEs for details.
