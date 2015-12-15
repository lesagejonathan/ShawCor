import ShawCor as sc
from numpy import array
import pickle

c = []
a = []
r = []

pr = pickle.load(open('/Users/jlesage/Dropbox/ShawCor/new_scans_for_jon/primer_plate_measurements.p','rb'))

for i in range(len(pr)):
	for j in range(5):

		try:
			xopt = sc.VelocityAttenuation(pr[i]['signals'][j],pr[i]['dt'],pr[i]['thickness'],2.9,[4,10],db=-30)

			c.append(xopt[0][1])
			a.append(xopt[0][0])
			r.append(xopt[1])
			
		except:

			pass


c = array(c)
a = array(a)
r = array(r)

PrimerVals = {'c':c,'alpha':a,'r':r}

pickle.dump(PrimerVals,open('/Users/jlesage/Dropbox/ShawCor/new_scans_for_jon/PrimerParameters.p','wb'))