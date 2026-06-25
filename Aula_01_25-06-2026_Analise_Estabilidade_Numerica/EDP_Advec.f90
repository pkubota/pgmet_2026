

MODULE Class_Fields
  IMPLICIT NONE
  PRIVATE
  INTEGER, PUBLIC      , PARAMETER  :: r8=8
  INTEGER, PUBLIC      , PARAMETER  :: r4=4
 REAL (KIND=r8) :: xMax=1
 REAL (KIND=r8) :: xMin=0
 REAL (KIND=r8),PUBLIC :: DeltaX=1.0
 REAL (KIND=r8),PUBLIC :: DeltaT=1.55  
 REAL (KIND=r8), PUBLIC :: C =0.2
! REAL (KIND=r8), PUBLIC :: ni = C*DeltaT/DeltaX
 INTEGER, PUBLIC :: Idim
 REAL (KIND=r8), PUBLIC :: xb0=100.0
 REAL (KIND=r8), PUBLIC :: xf0=400.0
 REAL (KIND=r8), PUBLIC :: tb0 =0
 REAL (KIND=r8), PUBLIC :: tf0 =0
 REAL (KIND=r8), PUBLIC :: xxb
 REAL (KIND=r8), PUBLIC :: yyf
 REAL (KIND=r8), PUBLIC :: Area

  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: xa(:) 
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: ua(:) 

  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: u(:) 
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: um(:) 
  REAL (KIND=r8), PUBLIC, ALLOCATABLE :: up(:) 

  PUBLIC :: Init_Class_Fields

CONTAINS  
!-----------------------------------------------------------------------------------------
  SUBROUTINE Init_Class_Fields()
    IMPLICIT NONE
      INTEGER :: i,xb(1),xf(1)
      REAL (KIND=r8):: t
      REAL (KIND=r8),ALLOCATABLE    :: diff(:)
      PRINT*,'DeltaX=',DeltaX,'DeltaT=',DeltaT,'CFL=',C*DeltaT/DeltaX
      Idim=100
      !Idim=  (xMax-Xmin)/DeltaX
      if (.not. allocated(u))  ALLOCATE (u(Idim))
      u=0.0
      if (.not. allocated(um))  ALLOCATE (um(Idim))
      um=0.0
      if (.not. allocated(up))  ALLOCATE (up(Idim))
      up=0.0

      if (.not. allocated(ua)) ALLOCATE (ua(Idim))
      ua=0.0
      if (.not. allocated(xa)) ALLOCATE (xa(Idim))
      if (.not. allocated(diff)) ALLOCATE (diff(Idim))

      DO i=1,Idim
         xa(i)=(i-1)*DeltaX
      END DO
      xb0=xa(Idim)/4.0
      xf0=xa(Idim)/2.0
      tb0 =0
      tf0 =0
      t=0
      xxb= xb0 + C*(t-tb0)
      yyf= xf0 + C*(t-tf0)
      DO i=1,Idim
         IF(xa(i) >xxb .and. xa(i) <yyf)THEN
            u(i)=1.0
         ELSE
            u(i)=0.0
         END IF
      END DO
      diff=ABS(xa-xxb)
      xb=MINLOC(diff) 
      diff=ABS(xa-yyf)
      xf=MINLOC(diff) 
      Area=( u(xf(1))-u(xb(1)-1))*(xa(xf(1))-xa(xb(1)))/(xf(1)-xb(1)+1)
      DO i=1,Idim
         IF(u(i) ==1.0)THEN
            u(i)=Area
         END IF
      END DO
      ua=u
      um=u
      up=u
  END SUBROUTINE Init_Class_Fields
!------------------------------------------------------------------------------------------
END MODULE Class_Fields



 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE Class_WritetoGrads
 USE Class_Fields, Only: Idim,xa
 IMPLICIT NONE
 PRIVATE
 INTEGER, PUBLIC      , PARAMETER  :: r8=8
 INTEGER, PUBLIC      , PARAMETER  :: r4=4
 INTEGER                    , PARAMETER :: UnitData=1
 INTEGER                    , PARAMETER :: UnitCtl=2
 CHARACTER (LEN=400)                   :: FileName
 LOGICAL                                            :: CtrlWriteDataFile
 PUBLIC :: SchemeWriteCtl
 PUBLIC :: SchemeWriteData
 PUBLIC :: InitClass_WritetoGrads
CONTAINS
 SUBROUTINE InitClass_WritetoGrads()
    IMPLICIT NONE
   FileName=''
   FileName='AdvecLinearConceitoErro'
   CtrlWriteDataFile=.TRUE.
 END SUBROUTINE InitClass_WritetoGrads

 FUNCTION SchemeWriteData(vars,irec)  RESULT (ok)
    IMPLICIT NONE
    REAL (KIND=r8), INTENT (INOUT) :: vars(Idim)
    INTEGER       , INTENT (INOUT) :: irec
    INTEGER        :: ok
    INTEGER        :: lrec
    REAL (KIND=r4) :: Yout(Idim)
    IF(CtrlWriteDataFile)INQUIRE (IOLENGTH=lrec) Yout
    IF(CtrlWriteDataFile)OPEN(UnitData,FILE=TRIM(FileName)//'.bin',&
    FORM='UNFORMATTED', ACCESS='DIRECT', STATUS='UNKNOWN', &
    ACTION='WRITE',RECL=lrec)
    ok=1
    CtrlWriteDataFile=.FALSE.
    Yout=REAL(vars(1:Idim),KIND=r4)
    irec=irec+1
    WRITE(UnitData,rec=irec)Yout
     ok=0
 END FUNCTION SchemeWriteData

 FUNCTION SchemeWriteCtl(nrec)  RESULT (ok)
    IMPLICIT NONE
    INTEGER, INTENT (IN) :: nrec
    INTEGER             :: ok,i
    ok=1
   OPEN(UnitCtl,FILE=TRIM(FileName)//'.ctl',FORM='FORMATTED', &
   ACCESS='SEQUENTIAL',STATUS='UNKNOWN',ACTION='WRITE')
    WRITE (UnitCtl,'(A6,A           )')'dset ^',TRIM(FileName)//'.bin'
    WRITE (UnitCtl,'(A                 )')'title  EDO'
    WRITE (UnitCtl,'(A                 )')'undef  -9999.9'
    WRITE (UnitCtl,'(A6,I8,A8   )')'xdef  ',Idim,' levels '
    WRITE (UnitCtl,'(10F16.10   )')(xa(i),i=1,Idim)
    WRITE (UnitCtl,'(A                  )')'ydef  1 linear  -1.27 1'
    WRITE (UnitCtl,'(A6,I6,A25   )')'tdef  ',nrec,' linear  00z01jan0001 1hr'
    WRITE (UnitCtl,'(A20             )')'zdef  1 levels 1000 '
    WRITE (UnitCtl,'(A           )')'vars 2'
    WRITE (UnitCtl,'(A           )')'uc 0 99 resultado da edol yc'
    WRITE (UnitCtl,'(A           )')'ua 0 99 solucao analitica ya'
    WRITE (UnitCtl,'(A           )')'endvars'
    CLOSE (UnitCtl,STATUS='KEEP')
    CLOSE (UnitData,STATUS='KEEP')
    ok=0
 END FUNCTION SchemeWriteCtl
END MODULE Class_WritetoGrads



 MODULE ModAdvection
  USE Class_Fields, Only: DeltaT,DeltaX,Idim,r8,xa,tf0,tb0,yyf,xxb,xb0,xf0,Area,C
   IMPLICIT NONE
   PRIVATE

  PUBLIC :: AnaliticFunction,Solve_Estabilidade_CTCS_2thCS,&
                             Solve_Estabilidade_FTCS_2thCS,&
                             Solve_Estabilidade_FTCS_3thCS,&
                             Solve_Estabilidade_FTCS_4thCS,&
                             Solve_Estabilidade_FTCS_2thCS2,&
                             Solve_Estabilidade_UpWind_1thBW ,&
			     Solve_Estavel_4thCS
                             

CONTAINS

!   
  FUNCTION AnaliticFunction(termX,ua,it)  RESULT (ok)

      REAL(KIND=r8), INTENT(INOUT) :: termX(Idim)
      REAL(KIND=r8), INTENT(IN   ) :: ua(Idim)
      INTEGER, INTENT(IN   ) :: it
      INTEGER          :: i2,xb,xc,xf,i
      INTEGER         :: ok
      REAL(KIND=r8)    :: t
      t=(it)*DeltaT

      yyf= xf0 + C*(t-tf0)
      IF(yyf >= xa(Idim))THEN
         xf0=0.0
         yyf=xf0
         tf0=t
      END IF 

      xxb= xb0 + C*(t-tb0)
      IF(xxb >= xa(Idim))THEN
         xb0=0.0
         xxb=xb0
         tb0=t
      END IF 
      IF(xf0 <= xb0 .and. yyf <= xxb) THEN  
         DO i=1,Idim
            IF(xa(i) > xxb )THEN
               termX(i)=Area
            ELSE IF( xa(i) < yyf )THEN
               termX(i)=Area
            ELSE
               termX(i)=0.0
            END IF
         END DO
      ELSE
         DO i=1,Idim
            IF(xa(i) > xxb .and. xa(i) < yyf)THEN
               termX(i)=Area
            ELSE
               termX(i)=0.0
            END IF
         END DO
      END IF
    ok=0
   END FUNCTION AnaliticFunction


  FUNCTION Solve_Estabilidade_UpWind_1thBW(termX,u)   RESULT (ok)
      REAL(KIND=r8), INTENT(INOUT) :: termX(Idim)
      REAL(KIND=r8), INTENT(IN   ) :: u(Idim)
      INTEGER :: i2,xb,xc,xf,i,ok
      DO i=1,Idim
         CALL index(i,xb,xc,xf)
         !termX(i) =u(xc) - C*DeltaT*(u(xc) - u(xb))/DeltaXC
         termX(i) =u(xc) - (C*(DeltaT/DeltaX))*(u(xc) - u(xb))
 
      END DO
   END   FUNCTION Solve_Estabilidade_UpWind_1thBW


  FUNCTION   Solve_Estabilidade_FTCS_2thCS2(termX,u)   RESULT (ok)
      REAL(KIND=r8), INTENT(INOUT) :: termX(Idim)
      REAL(KIND=r8), INTENT(IN   ) :: u(Idim)
      INTEGER :: i2,xb3,xb2,xb,xc,xf,xf2,xf3,i,ok

      DO i=1,Idim
         CALL index2(i,xb3,xb2,xb,xc,xf,xf2,xf3)
         !termX(i) = u(xc) -  C*DeltaT*( (u(xf) - u(xb) ))/(2.0*DeltaX)
         !termX(i) = u(xc) -  (C*(DeltaT/(6.0*DeltaX)))* &
         !       ( (-u(xc) + 2*u(xc) - u(xb)+ u(xb2)))
         !termX(i) = u(xc) -  (C*(DeltaT/(6.0*DeltaX)))* &
         !        ( (4.0*u(xb) - 3*u(xc) - u(xf2)))
         !termX(i) = u(xc) -  (C*(DeltaT)) * (&
         !          (    (u(xf2) - 9*u(xc) + 8*u(xb))/(10.0*DeltaX) ) + &
         !          (    (u(xf2) - 2*u(xc) + u(xb2))/(20.0*DeltaX) )    &
         !          )                
         termX(i) = u(xc) -  (C*(DeltaT)) * (&
                   ( (u(xf2)   - 9*u(xb) + 8*u(xb))/(6.0*DeltaX) )+ &
                   ( (u(xf2) - 2*u(xc) + u(xb2))/(4.0*DeltaX) )  &
                   )                

      END DO
    ok=0
  END FUNCTION  Solve_Estabilidade_FTCS_2thCS2

  FUNCTION   Solve_Estabilidade_FTCS_2thCS(termX,u)   RESULT (ok)
      REAL(KIND=r8), INTENT(INOUT) :: termX(Idim)
      REAL(KIND=r8), INTENT(IN   ) :: u(Idim)
      INTEGER :: i2,xb,xc,xf,i,ok
      DO i=1,Idim
         CALL index(i,xb,xc,xf)
         !termX(i) = u(xc) -  C*DeltaT*( (u(xf) - u(xb) ))/(2.0*DeltaX)
         termX(i) = u(xc) -  (C*(DeltaT/(2.0*DeltaX)))*( (u(xf) - u(xb) ))
      END DO
    ok=0
  END FUNCTION  Solve_Estabilidade_FTCS_2thCS

  FUNCTION   Solve_Estabilidade_FTCS_3thCS(termX,u)   RESULT (ok)
      REAL(KIND=r8), INTENT(INOUT) :: termX(Idim)
      REAL(KIND=r8), INTENT(IN   ) :: u(Idim)
      INTEGER :: i2,xb3,xb2,xb,xc,xf,xf2,xf3,i,ok
      DO i=1,Idim
         CALL index2(i,xb3,xb2,xb,xc,xf,xf2,xf3)
         !termX(i) = u(xc) -  C*DeltaT*( (u(xf) - u(xb) ))/(2.0*DeltaX)
         termX(i) = u(xc) -  (C*(DeltaT)) * (&
                   ( (u(xf2)   - 2*u(xc) + u(xb2))/(4.0*DeltaX) )+ &
                   ( (-4*u(xb) + 5*u(xc) - u(xf2))/(3.0*DeltaX) )  &
                   )                
      END DO
    ok=0
  END FUNCTION  Solve_Estabilidade_FTCS_3thCS


  FUNCTION   Solve_Estabilidade_FTCS_4thCS(termX,u)   RESULT (ok)
      REAL(KIND=r8), INTENT(INOUT) :: termX(Idim)
      REAL(KIND=r8), INTENT(IN   ) :: u(Idim)
      REAL(KIND=r8) :: mi=0.7
      INTEGER :: i2,xb3,xb2,xb,xc,xf,xf2,xf3,i,ok
      DO i=1,Idim
         CALL index2(i,xb3,xb2,xb,xc,xf,xf2,xf3)
         !termX(i) = u(xc) -  C*DeltaT*( (u(xf) - u(xb) ))/(2.0*DeltaX)
         !termX(i) = u(xc) -  (C*(DeltaT/(12.0*DeltaX)))* &
         !         ( (-u(xf2) + 8.0*u(xf) - 8.0*u(xb)+ u(xb2)))

         termX(i) = u(xc) -  (mi/12.0)* &
                  ( (-u(xf2) + 8.0*u(xf) - 8.0*u(xb)+ u(xb2)))

      END DO
    ok=0
  END FUNCTION  Solve_Estabilidade_FTCS_4thCS

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  FUNCTION   Solve_Estabilidade_CTCS_2thCS(termX,u)   RESULT (ok)
      REAL(KIND=r8), INTENT(INOUT) :: termX(Idim)
      REAL(KIND=r8), INTENT(IN   ) :: u(Idim)
      INTEGER :: i2,xb,xc,xf,i,ok
      DO i=1,Idim
         CALL index(i,xb,xc,xf)
         !termX(i) = u(xb) -  C*2*DeltaT*( (u(xf) - u(xb) ))/(2.0*DeltaX)
         termX(xc) = u(xb) -  ((C*DeltaT)/(2*DeltaX))*( (u(xf) - u(xb) ))
      END DO
    ok=0
  END FUNCTION  Solve_Estabilidade_CTCS_2thCS


  FUNCTION   Solve_Estavel_4thCS(termX,u)   RESULT (ok)
      REAL(KIND=r8), INTENT(INOUT) :: termX(Idim)
      REAL(KIND=r8), INTENT(IN   ) :: u(Idim)
      REAL(KIND=r8) :: dudx(Idim),du2dx2(Idim),du3dx3(Idim),du4dx4(Idim)
      INTEGER :: i2,xb3,xb2,xb,xc,xf,xf2,xf3,i,ok
      INTEGER :: test
      DO i=1,Idim
         CALL index2(i,xb3,xb2,xb,xc,xf,xf2,xf3)
!         dudx(xc)  = ( (u(xb2) - u(xf2)) + 8.0*(u(xf) - u(xb)))/ &
!                                ((12*DeltaX)) 

         dudx(xc) = ( (u(xc) - u(xb) ))/(DeltaX)

!         du2dx2(xc) = (-(u(xf2)+u(xb2)) + 16.0*u(xf) - 30.0*u(xc) + 16.0*u(xb)  )/&
!                                (12.0*(DeltaX**2))  

         du2dx2(xc) =(u(xf) - 2*u(xc) + u(xb))/(2*DeltaX*DeltaX)
 
         du3dx3(xc) = ((u(xf2) - u(xb2)) - 2.0*(u(xf) + u(xb)) )/&
                                (2.0*(DeltaX**3))  

         du4dx4(xc) = (u(xf2) - 4.0*u(xf) + 6.0*u(xc) - 4.0*u(xb) + u(xb2))/&
                      ((DeltaX**4)) 

         termX(xc) = u(xc) - (C*DeltaT)*( dudx(xc)) + ((DeltaT*DeltaT*C*C)/2.0)*(du2dx2(xc)) - &
                     (((DeltaT**3)*(C**3))/6.0)*( du3dx3(xc)) + (((DeltaT**4)*(C**4))/24.0)*( du4dx4(xc))

!         termX(xc) = u(xb) - (C*(2.0*DeltaT))*( dudx(xc)) + ((((2.0*DeltaT)**2)*C**2)/2.0)*(du2dx2(xc)) - &
!                     ((((2.0*DeltaT)**3)*(C**3))/6.0)*( du3dx3(xc)) + ((((2.0*DeltaT)**4)*(C**4))/24.0)*( du4dx4(xc))

      END DO

    ok=0
  END FUNCTION  Solve_Estavel_4thCS


!   
   SUBROUTINE index(i,xb,xc,xf)
      IMPLICIT NONE
      INTEGER, INTENT(IN   ) :: i
      INTEGER, INTENT(OUT  ) :: xb,xc,xf
      IF(i==1) THEN
        xb=Idim
        xc=i
        xf=i+1
      ELSE IF(i==Idim)THEN
        xb=Idim-1
        xc=Idim
        xf=1
      ELSE
        xb=i-1
        xc=i
        xf=i+1
      END IF
   END SUBROUTINE index

   SUBROUTINE index2(i,xb3,xb2,xb,xc,xf,xf2,xf3)
      IMPLICIT NONE
      INTEGER, INTENT(IN   ) :: i
      INTEGER, INTENT(OUT  ) :: xb3,xb2,xb,xc,xf,xf2,xf3
      IF(i==1) THEN
        xb3=Idim-2
        xb2=Idim-1
        xb=Idim
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      ELSE IF(i==2)THEN
        xb3=Idim-1
        xb2=Idim
        xb=i-1
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      ELSE IF(i==3)THEN
        xb3=Idim
        xb2=i-2
        xb=i-1
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      ELSE IF(i==Idim)THEN
        xb3=Idim-3
        xb2=Idim-2
        xb=Idim-1
        xc=i
        xf=1
        xf2=2
        xf3=3
      ELSE IF(i==Idim-1)THEN
        xb3=Idim-4
        xb2=Idim-3
        xb=Idim-2
        xc=i
        xf=Idim
        xf2=1
        xf3=2
      ELSE IF(i==Idim-2)THEN
        xb3=Idim-5
        xb2=Idim-4
        xb=Idim-3
        xc=i
        xf=Idim-1
        xf2=Idim
        xf3=1
      ELSE
        xb3=i-3
        xb2=i-2
        xb=i-1
        xc=i
        xf=i+1
        xf2=i+2
        xf3=i+3
      END IF
   END SUBROUTINE index2
 !------------------------------------------------------------------------------------------

END MODULE ModAdvection
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM  Main
  USE Class_Fields, Only : Init_Class_Fields,DeltaT,DeltaX,ua,u,um,up,r8,Idim,C
  USE ModAdvection, Only : AnaliticFunction,Solve_Estabilidade_CTCS_2thCS,&
                                            Solve_Estabilidade_UpWind_1thBW,&
                                            Solve_Estabilidade_FTCS_2thCS,&
                                            Solve_Estabilidade_FTCS_3thCS,&
                                            Solve_Estabilidade_FTCS_4thCS,&
                                            Solve_Estabilidade_FTCS_2thCS2,&
                                            Solve_Estavel_4thCS
  USE Class_WritetoGrads, Only : SchemeWriteCtl,SchemeWriteData,InitClass_WritetoGrads
   IMPLICIT NONE
   INTEGER, PARAMETER :: niter=300
   INTEGER :: irec_err,unit1,unit2,lrec

      CALL Init()
      CALL Run(irec_err,unit2)
      CALL Finalize()

  CONTAINS

  SUBROUTINE Init()
      CALL Init_Class_Fields()
      CALL InitClass_WritetoGrads
  END SUBROUTINE Init

  SUBROUTINE Run(irec_err,unit) 
      INTEGER, INTENT(INOUT) :: irec_err
      INTEGER, INTENT(IN   ) :: unit
      REAL (KIND=r8) :: termX(Idim)
      REAL (KIND=r8) :: termXm(Idim)
      REAL (KIND=r8) :: termXa(Idim)
      REAL (KIND=r8) :: err
      INTEGER :: i
      INTEGER :: it,lrec,irec,test
      irec=0
      err=0
      test=SchemeWriteData(u ,irec)
      test=SchemeWriteData(ua,irec)
      DO it=1,niter
        DO i=1,Idim
           termX(i)=0.0
           termXa(i)=0.0
        END DO  

        test=AnaliticFunction(termXa,ua,it)
!!        test=Solve_Estabilidade_UpWind_1thBW(termX,u)
!!        test=Solve_Estabilidade_FTCS_2thCS(termX,u)
!!        test=Solve_Estabilidade_FTCS_2thCS2(termX,u)
!!        test=Solve_Estabilidade_FTCS_3thCS(termX,u)

!        test=Solve_Estabilidade_FTCS_4thCS(termX,u)
!       test=Solve_Estabilidade_CTCS_2thCS(termX,u)
       test=Solve_Estavel_4thCS(termX,u)
	
        DO i=1,Idim
           um(i) = u(i)
           u(i) =termX(i)
           ua(i)=termXa  (i)
        END DO   
!        
        err=err+SUM((u-ua)**2)

        test=SchemeWriteData(u ,irec)
        test=SchemeWriteData(ua,irec)

      END DO
      test=SchemeWriteCtl(niter)

   PRINT*,'err=',err/niter,'DeltaX=',DeltaX,'DeltaT=',DeltaT,'CFL=',C*DeltaT/DeltaX

  END SUBROUTINE Run
!     
  SUBROUTINE Finalize()
      
  END SUBROUTINE Finalize
END PROGRAM  Main
