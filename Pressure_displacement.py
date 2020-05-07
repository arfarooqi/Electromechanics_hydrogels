from __future__ import print_function
from dolfin import (File, Constant, near, IntervalMesh, interval, split, MeshFunction, cells, refine, CompiledSubDomain, DirichletBC, Point, FiniteElement, \
                    MixedElement, FunctionSpace, Function, UserExpression, TestFunctions, derivative, NonlinearVariationalProblem, assign, dot, ds, \
                    inner, grad, dx, SubMesh, solve, plot, TestFunction, TrialFunction, VectorFunctionSpace, as_vector, VectorElement, project, FacetNormal, \
                    interpolate, Expression, NonlinearVariationalSolver, nabla_grad, TrialFunctions, assemble, LinearVariationalSolver, RectangleMesh, \
                    LinearVariationalProblem, parameters, HDF5File, Mesh)
#from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
parameters["allow_extrapolation"] = True

from timeit import default_timer as timer
startime = timer() 

plt.rc('font', family='Arial')
import matplotlib
matplotlib.font_manager._rebuild()
plt.rcParams["mathtext.fontset"] = "dejavuserif"
###############################################################################
Temp = Constant(298.0)
R = Constant(8.3143)
E_gel = 124E3

#this is how you load
###############################################################################
mesh = Mesh('mesh.xml.gz')

f = HDF5File(mesh.mpi_comm(), 'Field_solution.h5', 'r') 
P_gel = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_gel = MixedElement([P_gel, P_gel, P_gel])

ME_new = FunctionSpace(mesh, element_gel)
u_sol = Function(ME_new)
f.read(u_sol,'solution_full')
an_gel, ca_gel, psi_gel = u_sol.split(deepcopy = True)

f.close()

PNP_time = timer()
###############################################################################
x = np.linspace(5.0E-3, 10.0E-3, 1000) 
y = np.linspace(5.01E-3, 9.99E-3, 1000)  
z = np.linspace(0.0, 15.0E-3, 1000)

an_gel_line = np.array([an_gel(point1) for point1 in z])
ca_gel_line = np.array([ca_gel(point1) for point1 in z])

subdomain = CompiledSubDomain("x[0] >= 5.0e-3 && x[0] <= 10.0e-3")
subdomains = MeshFunction("size_t", mesh, 1)
subdomains.set_all(0)  
subdomain.mark(subdomains, 1)

mesh_gel = SubMesh(mesh, subdomains, 1)

############################  Osmotic pressure  ###############################
ca_tel = MeshFunction("size_t", mesh_gel, 1)
an_tel = MeshFunction("size_t", mesh_gel, 1)

x1 = 5.0E-3
x2 = 10.0E-3
x3 = 5.0E-3
x4 = 10.0E-3

V_osm = FunctionSpace(mesh_gel, P_gel)

an_tel = interpolate(Expression("(an_x2 - an_x1) / (x2 - x1) * (x[0] - x1) + an_x1", degree=2, an_x1=an_gel(x1), an_x2=an_gel(x2), x2=x2, x1=x1), V_osm)
ca_tel = interpolate(Expression("(ca_x4 - ca_x3) / (x4 - x3) * (x[0] - x3) + ca_x3", degree=2, ca_x3=ca_gel(x3), ca_x4=ca_gel(x4), x4=x4, x3=x3), V_osm)
###############################################################################

an_tel_line = np.array([an_tel(point) for point in x])
ca_tel_line = np.array([ca_tel(point) for point in x])

plt.plot(z, an_gel_line, color = 'k', linestyle = '-', linewidth = 1.5)
plt.plot(z, ca_gel_line, color = 'b', linestyle = '--', linewidth = 1.5)
plt.grid(color = 'k', linestyle = '-', linewidth = 0.1)
plt.xlabel('Length in $x$-direction [m]')
plt.ylabel('Ion Concentrations [mM]')
plt.legend(['$c^-$', '$c^+$', r'$\tilde{c}^-$', r'$\tilde{c}^+$'], loc='best', markerfirst = False, prop={'size': 15})
plt.xlim(0.0, 0.015)
plt.xticks([0, 0.005, 0.010, 0.015])
plt.savefig('all_compar.png', dpi=800, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, \
            bbox_inches='tight', pad_inches=0.1, frameon=None)
plt.show()

const = R*Temp
p_int = interpolate(Expression('x[0] >= 5.0E-3 && x[0] <= 10.0E-3 ? 2052.488 : 0.0', degree = 1), V_osm)
p_osm1 = project((const*(an_gel - an_tel + ca_gel - ca_tel)), V_osm)
p_osm2 = project((const*(an_gel - an_tel + ca_gel - ca_tel) - p_int), V_osm)
     
p_osm_line = np.array([p_osm1(point1) for point1 in y])

plt.plot(y, p_osm_line*1E-3, 'k-', linewidth=1.5)
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.legend(['Osmotic pressure'], loc='best', markerfirst = False, prop={'size': 15})
plt.ylabel('Osmotic pressure [kPa]')
plt.xlabel('Length in $x$-direction [m]')
plt.ylim(1.0, 2.6)
plt.savefig('pressure.png', dpi=800, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, \
            bbox_inches='tight', pad_inches=0.1, frameon=None)
plt.show()

OP_time = timer()
##########################  END Osmotic pressure  #############################

##############################   Diplacement  #################################
def fix_point(x, on_boundary):
    return on_boundary and near(x[0], 5E-3)

ME1 = FunctionSpace(mesh_gel, 'P', 1)

bc_fix = DirichletBC(ME1, Constant(0.0), fix_point)

u_d = TrialFunction(ME1)
v_d = TestFunction(ME1)

F_d = -E_gel*inner(grad(u_d), grad(v_d))*dx #+ E_gel*inner(grad(u_d), n)*v_d*ds
L_d = inner(grad(p_osm2)[0], v_d)*dx 

u_d = Function(ME1)

solve(F_d == L_d, u_d, bc_fix)

u_d_line = np.array([u_d(point1) for point1 in y])
fig = plt.figure(figsize=(6, 4))
plt.plot(y, 1e3*u_d_line, color='k', linestyle='-', linewidth=1.5)
ax = plt.gca()
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.xlabel('Length in $x$-direction [m]')
plt.ylabel('Displacement [mm]')
ax.get_yaxis().get_major_formatter().set_useOffset(False)

x_axis1 = [0.00138, 0.09229, 0.19146, 0.30165, 0.40634, 0.51653, 0.61019, 0.70386, 0.79752, 0.88567, 0.99036, 1.08953, 1.19972, 1.30441, 1.41460, 1.49725, 
           1.61295, 1.74518, 1.82231, 1.93251, 2.03168, 2.13085, 2.23003, 2.34573, 2.44490, 2.54959, 2.64325, 2.73691, 2.81956, 2.93526, 3.03444, 3.12810, 
           3.22727, 3.31543, 3.41460, 3.50826, 3.57989, 3.66253, 3.73967, 3.82231, 3.90496, 4.02617, 4.10882, 4.19146, 4.27410, 4.34573, 4.45592, 4.51653, 
           4.60468, 4.67631, 4.75344, 4.84160, 4.91873, 4.99587]
x_axis_51 = [0.005 + (x*1e-3) for x in x_axis1]
y_axis1 = [0.00022, 0.00130, 0.00247, 0.00384, 0.00521, 0.00658, 0.00775, 0.00912, 0.01039, 0.01156, 0.01303, 0.01450, 0.01587, 0.01743, 0.01899, 0.02027, 
           0.02193, 0.02398, 0.02515, 0.02691, 0.02838, 0.02994, 0.03151, 0.03327, 0.03483, 0.03679, 0.03835, 0.03992, 0.04128, 0.04324, 0.04500, 0.04666, 
           0.04832, 0.04989, 0.05184, 0.05351, 0.05478, 0.05624, 0.05771, 0.05918, 0.06074, 0.06318, 0.06475, 0.06631, 0.06797, 0.06934, 0.07149, 0.07267, 
           0.07453, 0.07599, 0.07746, 0.07932, 0.08088, 0.08235]
y_axis1_1 = [0 + x for x in y_axis1]
plt.plot(x_axis_51, y_axis1_1, color='b', linestyle='-.', linewidth=1.5)
plt.legend(['Current work', 'LiHua2009'], loc='best', markerfirst = False, prop={'size': 15})
plt.savefig('displacement.png', dpi=800, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, \
            bbox_inches='tight', pad_inches=0.1, frameon=None)
plt.show()
############################  END Diplacement  ################################

aftersolveT = timer() 

aftersolveT = timer() 
totime_PNP = PNP_time - startime
totime_OP = OP_time - PNP_time
totime = aftersolveT - startime

print("Number of DOFs for PNP system : {}".format(ME_new.dim()))
print("Number of DOFs for OP : {}".format(V_osm.dim()))

print("Total time of simulation for PNP : " + str(round(totime_PNP)))
print("Total time of simulation for OP : " + str(round(totime_OP)))
print("Total time of complete simulation : " + str(round(totime)))