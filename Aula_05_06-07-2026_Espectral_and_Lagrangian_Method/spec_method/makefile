#
#  $Author: pkubota $
#  $Date: 2007/10/10 20:24:23 $
#  $Revision: 1.1 $
#
# $Id: Makefile.linux64,v 1.1 2007/10/10 20:24:23 pkubota Exp $
#
#
#  PC Cluster, mpi scali
#
FTRACE=  
OPENMP= 
F90=gfortran -ffree-form  -ffree-line-length-none 
AR=ar
F90_32FLAGS = 
F90_32BITS = 
F77=gfortran  -ffree-form -ffree-line-length-none 
FFLAGS = 
CC     =gcc   
flagOpt="-O2"
flag64bit= 
MPIF90=mpif90 -ffree-form  -ffree-line-length-none

TRACE= 
OPENMP=
F90_32FLAGS = 
F90_32BITS = $(FTRACE) $(OPENMP) 
LOADFLAG= 


# OPTIONS FOR POS
CPPFLAGS = -D_UNDERSCORE -DLINUX
FFLAGS	= 
CFLAGS	= -w
ARFLAGS	= 
CC      =gcc   
# If there is a version difference between your gcc and the gcc
# used to compile the ifort binary, you may have to add a flag
# like -gcc-version=400 do CFLAG
LOADFLAG= 
# Bibliotecas necessarias
#LIBS = -leccodes_f90 -leccodes
#

#
###include Makefile.common
#
EXEC=advection_spectral.x

#ECCODES_SAMPLES_PATH = /mnt/beegfs/paulo.kubota/lib/eccodes/share/eccodes/samples
#ECCODES_INCLUDE = -I/mnt/beegfs/paulo.kubota/lib/eccodes/include
#ECCODES_LIB = -L/mnt/beegfs/paulo.kubota/lib/eccodes/lib $(LIBS)

OUTPUTINC = -I./dfftpack
OUTPUTLIB = -L./dfftpack -ldfftpack

LINKOPTS  = -ldfftpack
LIB_NETCDF= $(OUTPUTLIB)  $(OUTPUTINC)

.SUFFIXES: .o .f90 .F90 .c 
#
# *****************************************************************
#
#OBJS_CC = advection_spectral.o

#OBJS = 	advection_spectral.o 
#
SRC	= $(OBJ_MOD:.o=.f90) $(OBJS:.o=.f90) $(OBJS_CC:.o=.c)

$(EXEC):  advection_spectral.f90 
	$(F90) -o $(EXEC) $(F90FLAG) $(LOADFLAG) $(LIB_NETCDF)  advection_spectral.f90 dfftpack/libdfftpack.a


clean:
	-rm -f $(OBJS)
	-rm -f $(OBJS_CC)
	-rm -f $(EXEC)
	-rm -f *.mod

.F90.o :
	$(F90) $(F90FLAG) -c  $<

.f90.o:
	$(F90) $(F90FLAG) -c $< 
.c.o:
	$(CC) $(CFLAGS) $(DEFS) -c $< 
 
