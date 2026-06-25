'reinit'
'set display color white'
'c'
'open AdvecLinearConceitoErro.ctl'

it=1
while(it<=200)

'd ua(t='it')'
'd uc(t='it')'
'!sleep 1'
'c'
it=it+1
endwhile
