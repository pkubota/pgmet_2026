'reinit'
'set display color white'
'c'
'open GravityWaveConceitual1D.ctl'
*'open AdvecLinearConceitual1D.ctl'

it=0
while(it<=2400)
'c'
'd ua(t='it')'
'd uc(t='it')'
if(it=2399)
  'q pos'  
  it=0
endif
it=it+1
endwhile
