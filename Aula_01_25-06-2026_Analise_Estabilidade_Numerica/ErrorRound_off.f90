PROGRAM Main
  IMPLICIT NONE
  INTEGER, PARAMETER :: single_kind = 4!selected_real_kind( p=6 , r=37 )
  INTEGER, PARAMETER :: double_kind = 8!selected_real_kind( p=15, r=200 )  
  INTEGER, PARAMETER :: N=1e8
  REAL (KIND=single_kind) :: s_X
  REAL (KIND=single_kind) :: s_X_m
  REAL (KIND=double_kind) :: d_X
  INTEGER      :: i 
  s_X=0.0_single_kind;d_X=0.0_double_kind
  DO i=1, N
    s_X_m=s_X

    s_X = s_X + 1.0_single_kind
    d_X = d_X + 1.0_double_kind
    !IF(s_X == s_X_m)print*,s_X,d_X
  END DO
  PRINT*,'soma_s_X=',s_X,'media_s_X=',s_X/N
  PRINT*,'soma_d_X=',d_X,'media_d_X=',d_X/N
END PROGRAM Main

