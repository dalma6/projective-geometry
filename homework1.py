import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

points = {}
matrices_2x9 = {}

# help functions

def initialization(ind):
	points['A'] = (-10, 4)
	points['B'] = (-7, 5)
	points['C'] = (-2, 1)
	points['D'] = (-9, 3)
	points['E'] = (-8, 1)

	points['Ap'] = (5, 5)
	points['Bp'] = (0, 3)
	points['Cp'] = (2, 1)
	points['Dp'] = (8, 1)
	points['Ep'] = (4, 6)
	
	if ind == 1:
		return

	# homogenize points
	for point in points:
		points[point] += (1,)
		
def create_2x9_matrix(X, Xp):
	return [[0, 0, 0, -Xp[2]*X[0], -Xp[2]*X[1], -Xp[2]*X[2], Xp[1]*X[0], Xp[1]*X[1], Xp[1]*X[2]],
				  [Xp[2]*X[0], Xp[2]*X[1], Xp[2]*X[2], 0, 0, 0, -Xp[0]*X[0], -Xp[0]*X[1], -Xp[0]*X[2]]]

def normalization(points):
	bX = sum([x[0] for x in points])/len(points)
	bY = sum([x[1] for x in points])/len(points)
    
	translation_matrix = np.array([[1, 0, -bX],
								   [0, 1, -bY],
								   [0, 0, 1]])
	
	mean_distance = sum(list(map(lambda x: np.sqrt((x[0] - bX)**2 + (x[1] - bY)**2), points)))/len(points)
	scale_par = np.sqrt(2)/mean_distance
	homothety_matrix = np.array([[scale_par, 0, 0], 
							  [0, scale_par, 0], 
							  [0, 0, 1]])
    
	T = np.dot(translation_matrix, homothety_matrix)
	return T

# algorithms

def naive():
	initialization(0)
	
	base_original = np.transpose([points['A'], points['B'], points['C']])
	solution_eq_system_original = np.linalg.solve(base_original, np.transpose(points['D']))
	
	base_projected = np.transpose([points['Ap'], points['Bp'], points['Cp']])
	solution_eq_system_projected = np.linalg.solve(base_projected, np.transpose(points['Dp']))
	P1 = solution_eq_system_original * base_original 
	P2 = solution_eq_system_projected * base_projected
	
	try:
		P1_inv = np.linalg.inv(P1)
	except ArithmeticException:
		raise LinAlgError('Singular matrix')
	
	global P_naive 
	P_naive = np.dot(P2, P1_inv)

#	print 'Naivni'
#	print (P_naive)
	
def DLT(points, ind):
	if ind == 0:
		initialization(0)
		m1 = create_2x9_matrix(points['A'], points['Ap'])
		m2 = create_2x9_matrix(points['B'], points['Bp'])
		m3 = create_2x9_matrix(points['C'], points['Cp'])	
		m4 = create_2x9_matrix(points['D'], points['Dp'])
		m5 = create_2x9_matrix(points['E'], points['Ep'])
	else:
		m1 = create_2x9_matrix(points[0], points[5])
		m2 = create_2x9_matrix(points[1], points[6])
		m3 = create_2x9_matrix(points[2], points[7])	
		m4 = create_2x9_matrix(points[3], points[8])	
		m5 = create_2x9_matrix(points[4], points[9])
	
	final_2x9 = np.concatenate((m1, m2, m3, m4, m5), axis=0)
	#print final_2x9
	
	U, D, Vt = np.linalg.svd(final_2x9)
	P = Vt[-1][:].reshape((3, 3))
	P = P_naive[0, 0] / P[0, 0] * P
	
	if ind == 0:
		print 'DLT'
		print(P)
	return P
	
def normalized_DLT():
	initialization(0)

	originals = [points['A'], points['B'], points['C'], points['D'], points['E']]
	projections = [points['Ap'], points['Bp'], points['Cp'], points['Dp'], points['Ep']]
	
	originals_t = normalization(originals)
	A_norm = np.dot(originals_t, points['A'])
	B_norm = np.dot(originals_t, points['B'])
	C_norm = np.dot(originals_t, points['C'])
	D_norm = np.dot(originals_t, points['D'])
	E_norm = np.dot(originals_t, points['E'])
	
	projections_t = normalization(projections)
	Ap_norm = np.dot(projections_t, points['Ap'])
	Bp_norm = np.dot(projections_t, points['Bp'])
	Cp_norm = np.dot(projections_t, points['Cp'])
	Dp_norm = np.dot(projections_t, points['Dp'])
	Ep_norm = np.dot(projections_t, points['Ep'])
	
	pts = [A_norm, B_norm, C_norm, D_norm, E_norm, Ap_norm, Bp_norm, Cp_norm, Dp_norm, Ep_norm]
	P_dlt = DLT(pts, 1)
	
	P = np.linalg.inv(projections_t).dot(P_dlt).dot(originals_t)
	P = P_naive[0, 0] / P[0, 0] * P
	print 'Modifikovani DLT'
	print(P)
	return P

# additional
def draw():
	initialization(1)
	originals = [list(points['A']), list(points['B']), list(points['C']), list(points['D']), list(points['E'])]
	projections = [list(points['Ap']), list(points['Bp']), list(points['Cp']), list(points['Dp']), list(points['Ep'])]
	
	p = Polygon(originals, fill=False, color='red')
	ax = plt.gca()
	ax.add_patch(p)
	p2 = Polygon(projections, fill=False)
	ax2 = plt.gca()
	ax2.add_patch(p2)
	ax.set_xlim(-10,10)
	ax.set_ylim(-10,10)
	
	plt.show()
	
def main():
	naive()
	DLT(points, 0)
	normalized_DLT()
	draw()
	
if __name__ == '__main__':
	main()