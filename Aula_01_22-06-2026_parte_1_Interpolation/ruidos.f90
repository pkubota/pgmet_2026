Module ModClassDerivada
 IMPLICIT NONE
 INTEGER                  :: xdim
 REAL, ALLOCATABLE        :: f(:)
 REAL, ALLOCATABLE        :: ff(:)
 REAL, ALLOCATABLE        :: x(:)
 REAL, ALLOCATABLE        :: df2dx(:)
 REAL   :: dx
 REAL   :: Length
 REAL, PARAMETER   :: Pi=3.14
 PUBLIC :: InitClass,runDeriv,finalizeClassDerivada
CONTAINS

SUBROUTINE InitClass
 INTEGER :: i,xb,xc,xf
 Length=2*Pi
 xdim  =100
 Dx= Length/REAL(xdim)
 allocate(ff(0:xdim)); ff(0:xdim) =0.0
 allocate(df2dx(0:xdim)); df2dx(0:xdim) =0.0
 allocate(f(0:xdim)); f(0:xdim) =0.0
 allocate(x(0:xdim)); x(0:xdim) =0.0

 DO i=1,xdim
    CALL index(i,xdim,xb,xc,xf)
    x(xc) = x(xb) + Dx
 END DO

 DO i=0,xdim
    CALL index(i,xdim,xb,xc,xf)
    f (xc) = sin(x(xc))
    ff(xc) = cos(x(xc))
 END DO

END SUBROUTINE InitClass

SUBROUTINE runDeriv()
 INTEGER :: i,xb,xc,xf
 DO i=0,xdim
    CALL index(i,xdim,xb,xc,xf)
    ! df2dx(i) =  (( F(xf) - F(xb) )/(2.0*Dx))
     df2dx(i) =  (( F(xf) - F(xc) )/(Dx))
    !df2dx(i) =  (( F(xb) - F(xc) )/(Dx))
 END DO
  PRINT*,'Effective Error=',SUM(df2dx(1:xdim-1)-ff(1:xdim-1))/SUM(ff(1:xdim-1))
 CALL WriteBinary()
 CALL WriteCTL()
END SUBROUTINE runDeriv

SUBROUTINE WriteBinary()
  INTEGER :: lrec
  INQUIRE(IOLENGTH=lrec)df2dx(1:xdim-1)
  OPEN(1,FILE='runDeriv.bin',ACCESS='DIRECT',FORM='UNFORMATTED', RECL=lrec,&
       STATUS='UNKNOWN',ACTION='WRITE')
  WRITE(1,rec=1)df2dx(1:xdim-1)
  WRITE(1,rec=2)ff   (1:xdim-1)
  WRITE(1,rec=3)f    (1:xdim-1)
  CLOSE(1,STATUS='KEEP')
END SUBROUTINE WriteBinary

SUBROUTINE WriteCTL()
 INTEGER :: i
  OPEN(1,FILE='runDeriv.ctl',ACCESS='SEQUENTIAL',FORM='FORMATTED', &
       STATUS='UNKNOWN',ACTION='WRITE')

  WRITE(1,'(A6,A12         )')'dset ^','runDeriv.bin'
  WRITE(1,'(A              )')'*options big_endian'
  WRITE(1,'(A6,A10         )')'undef ','-999.99999'
  WRITE(1,'(A6,I5,A8       )')'xdef  ',xdim-1,' levels '
  WRITE(1,'(10F15.5)')(x(i),i=1,xdim-1)
  WRITE(1,'(A6,A3,A8,A7 ,A4)')'ydef  ',' 1 ',' linear ',' -90.0 ',' 1.0'
  WRITE(1,'(A6,A3,A8,A14,A4)')'tdef  ',' 1 ',' linear ',' 00z01apr2014 ',' 1hr'
  WRITE(1,'(A6,A3,A8,A4    )')'zdef  ',' 1 ',' levels ','1000'
  WRITE(1,'(A6,A3          )')'VARS  ',' 3 '
  WRITE(1,'(A6,A23         )')'Fnum  ','0 99 derivada numerica '
  WRITE(1,'(A6,A23         )')'Fana  ','0 99 derivada analitica'
  WRITE(1,'(A6,A23         )')'Fori  ','0 99 funcao original'
  WRITE(1,'(A7             )')'ENDVARS'
END SUBROUTINE WriteCTL

SUBROUTINE index(i,Idim,xb,xc,xf)
   IMPLICIT NONE
   INTEGER, INTENT(IN	) :: i
   INTEGER, INTENT(IN	) :: Idim
   INTEGER, INTENT(OUT  ) :: xb,xc,xf
   IF(i==0) THEN
     xb=Idim
     xc=i
     xf=i+1
   ELSE IF(i==Idim)THEN
     xb=Idim-1
     xc=Idim
     xf=0
   ELSE
     xb=i-1
     xc=i
     xf=i+1
   END IF
END SUBROUTINE index
SUBROUTINE finalizeClassDerivada()
   DEALLOCATE(F)
   DEALLOCATE(FF)
   DEALLOCATE(df2dx)
   DEALLOCATE(x)

END SUBROUTINE finalizeClassDerivada

END Module ModClassDerivada
!======================================================
PROGRAM Main
  USE ModClassDerivada, Only: InitClass,runDeriv,&
  finalizeClassDerivada
 IMPLICIT NONE

 CALL Init()
 CALL run()
 CALL finalize()
CONTAINS

SUBROUTINE Init()
    CALL InitClass()
END SUBROUTINE Init

SUBROUTINE run()
  CALL runDeriv()
END SUBROUTINE run

SUBROUTINE finalize()
  CALL finalizeClassDerivada()
END SUBROUTINE finalize

END PROGRAM Main
!======================================================
