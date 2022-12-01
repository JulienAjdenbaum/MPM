import numpy as np
import global_variables as gv

a = [[0.00364926, 0.00074657, 0.0077796 ],
 [0.00074657, 0.06402861, 0.00019191],
 [0.0077796,  0.00019191, 0.07551701]]
print(np.sqrt(np.linalg.eig(np.linalg.inv(a))[0])*gv.resolution*2*np.sqrt(2*np.log(2)))
