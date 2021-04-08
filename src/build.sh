# Build the model
# Only tested with GNU (gfortran)

dos2unix *.FOR

patch SCREEN3A.FOR SCREEN3A.FOR.patch
patch DEPVAR.INC DEPVAR.INC.patch

gfortran -cpp SCREEN3A.FOR SCREEN3B.FOR SCREEN3C.FOR -o SCREEN3.exe
