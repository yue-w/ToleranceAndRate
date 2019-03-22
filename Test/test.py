import functions as f

import numpy as np
a = np.array([1,2,3,4,5])
a.reshape(5,1)
np.savetxt('test.csv', a, delimiter=' ')
