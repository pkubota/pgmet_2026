'reinit'
'set display color white'
'c'
'open saida_GridA.ctl'
it=1
while(it<=3000   )
'rgbset_local.gs'
'clear'
'set gxout shaded'
'set ccolor 1'
'set clevs     0.1    0.2   0.3    0.4    0.5    0.6    0.7   0.8  0.9   1.0000   '
'set ccols 0     21     22     23     24     25     26    27     28   29        39'
'd y'
'set gxout contour'
'set clevs    0.996   0.9965   0.997    0.9975   0.998     0.999  0.9995     1.0000  1.0005 1.001 '
'set clevs  -0.0035   -0.0015  -0.0005  -0.0001   -0.00005    -0.00001  0.0001   0.001  0.007 0.015 '
'set ccols 21      22        24       26       28        29      49        36        35     34      32  31    '
'set clab off'
'd h(t='it')'
'!sleep 1'
say it
it=it+20
endwhile
