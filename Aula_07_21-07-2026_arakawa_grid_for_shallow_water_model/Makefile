#
# Targets are the suffixes of the system-specific makefiles in
# the makefiles/ directory.
# For example, to build PosGrib for Solaris, give the command
#
#   make solaris
#
# This builds an intermediate library in the util/ directory,
# then builds the nedit and nc executables in the source/ directory.
#
SHELL=/bin/sh
#
#  $Author: pkubota $
#  $Date: 2007/10/10 20:28:03 $
#  $Revision: 1.4 $
#
#  $Author: pkubota $
#  $Date: 2007/10/10 20:24:23 $
#  $Revision: 1.1 $
#
# $Id: Makefile.linux,v 1.1 2007/10/10 20:24:23 pkubota Exp $
#

#
# INTEL COMPILER - WORKING
#
# Sometimes ifort fails to statically link the executable with
# optmization giving and error message like:
#   ifort: error: problem during multi-file optimization compilation (code 1)
# Here we adopt the safe procedure of passing -O0 to LOADFLAG
#  
FFLAGS_DEBUG = -g  -fbounds-check -Waliasing -Wall -fbacktrace -ffpe-trap=invalid,overflow,zero  -finit-real=nan -finit-integer=nan  -ffree-line-length-none  -O0 -Warray-bounds  -ffast-math -funroll-loops -ftree-vectorizer-verbose=2
AR=ar
ARFLAGS = 
F90=gfortran  $(FFLAGS_DEBUG)
F90_32FLAGS = 
F90_32BITS = 
F77 =gfortran 
FFLAGS = 
CC =gcc 
# If there is a version difference between your gcc and the gcc
# used to compile the ifort binary, you may have to add a flag
# like -gcc-version=400 do CFLAG
CFLAGS =  -DLINUX 
LOADFLAG=
#
# $Id: Makefile.common,v 1.4 2007/10/10 20:28:03 pkubota Exp $
#
# Platform independent part of make procedure for POSGRIB directory, 
# included by machine specific makefiles.
#
EXEC=SWE_GRID_A.exe

OBJ = SWE_GRID_A.o

Main:	$(OBJ)
	$(F90) -o $(EXEC) $(F90_32BITS) $(LOADFLAG) $(OBJ) $(LIB)

SWE_GRID_A_Pedro.o: SWE_GRID_A.f90
	$(F90) -c $(F90_32BITS) SWE_GRID_A.f90

.SUFFIXES:
.SUFFIXES: .f90 .o


.f90.o:
	$(F90) -c $(F90_32BITS) $<

clean:
	-rm -f $(OBJ)
	-rm -f $(EXEC)
	-rm -f *.mod
	-rm -f *.o
