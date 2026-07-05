'reinit'
'set display color white'
'c'
*'open AdvecLinearConceitual1D_RK4CS4.ctlAdvecLinearConceitual1D.ctl'
'open AdvecLinearConceitual1D.ctl'

it=0 
while(it<=3000)
'c'
'd ua(t='it')'
'd uc(t='it')'

if(it=2999)
  'q pos'  
  it=0
endif
it=it+1
endwhile
