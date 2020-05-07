from __future__ import print_function
from dolfin import (File, Constant, near, IntervalMesh, interval, split, MeshFunction, cells, refine, CompiledSubDomain, DirichletBC, Point, FiniteElement, \
                    MixedElement, FunctionSpace, Function, UserExpression, TestFunctions, derivative, NonlinearVariationalProblem, interpolate, assign, \
                    inner, grad, dx, SubMesh, solve, plot, TestFunction, TrialFunction, VectorFunctionSpace, as_vector, VectorElement, project, FacetNormal, \
                    NonlinearVariationalSolver, Expression, nabla_grad, dot, ds, HDF5File)
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
startime = timer() 

D_an = Constant(1.0E-7) # m^2/s
D_ca = Constant(1.0E-7) # m^2/s
z_an = Constant(-1.0)
z_ca = Constant(1.0)
z_fc = Constant(-1.0)
Farad = Constant(9.6487E4) # C/mol
eps0 = Constant(8.854E-12) # As/Vm
epsR = Constant(80.0)
Temp = Constant(298.0)
R = Constant(8.3143)
E_gel = 120E3

################################## mesh part ##################################
mesh = IntervalMesh(7200000, 0.0, 15.0E-3) # 21.6 million

refinement_cycles = 3
for _ in range(refinement_cycles):
    refine_cells = MeshFunction("bool", mesh, 1)
    refine_cells.set_all(False)
    for cell in cells(mesh):
        if abs(cell.distance(Point(5.0e-3))) < 0.2e-3:
            refine_cells[cell] = True
        elif abs(cell.distance(Point(10.0e-3))) < 0.2e-3:
            refine_cells[cell] = True
    mesh = refine(mesh, refine_cells)

mesh_file = File("mesh.xml.gz")    
mesh_file << mesh

P1 = FiniteElement('P', interval, 1)
element = MixedElement([P1, P1, P1])
ME = FunctionSpace(mesh, element)

def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0)
def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], 15.0E-3)

subdomain = CompiledSubDomain("x[0] >= 5.0E-3 && x[0] <= 10.0E-3")
subdomains = MeshFunction("size_t", mesh, 1)
subdomains.set_all(0)        
subdomain.mark(subdomains, 1)

fc = Constant(2.0) # FCD
V0_r = FunctionSpace(mesh, 'DG', 0)
fc_function = Function(V0_r)
fc_val = [0.0, fc]
help = np.asarray(subdomains.array(), dtype = np.int32)
fc_function.vector()[:] = np.choose(help, fc_val)

Poten = 100.0E-3
Sol_c = Constant(1.0)
l_bc_an = DirichletBC(ME.sub(0), Constant(Sol_c), left_boundary)
r_bc_an = DirichletBC(ME.sub(0), Constant(Sol_c), right_boundary)
l_bc_ca = DirichletBC(ME.sub(1), Constant(Sol_c), left_boundary)
r_bc_ca = DirichletBC(ME.sub(1), Constant(Sol_c), right_boundary)
l_bc_psi = DirichletBC(ME.sub(2), Constant(-Poten), left_boundary)
r_bc_psi = DirichletBC(ME.sub(2), Constant(Poten), right_boundary)
bcs = [l_bc_an, r_bc_an, l_bc_ca, r_bc_ca, l_bc_psi, r_bc_psi]

u = Function(ME)
#######################  Initial properties assignment  #######################
V = FunctionSpace(mesh, P1)
an_int = interpolate(Expression('x[0] >= 5.0E-3 && x[0] <= 10.0E-3 ? 0.4142 : 1.0', degree = 1), V)
ca_int = interpolate(Expression('x[0] >= 5.0E-3 && x[0] <= 10.0E-3 ? 2.4142 : 1.0', degree = 1), V)
psi_int = interpolate(Expression('x[0] >= 5.0E-3 && x[0] <= 10.0E-3 ? -22.6322E-3 : 0.0', degree = 1), V)
assign(u.sub(0), an_int)
assign(u.sub(1), ca_int)
assign(u.sub(2), psi_int)
###############################################################################
an, ca, psi = split(u)

van, vca, vpsi = TestFunctions(ME)

Fan = D_an*(inner(grad(an), grad(van))*dx + (Farad / R / Temp * z_an*an)*inner(grad(psi), grad(van))*dx)
Fca = D_ca*(inner(grad(ca), grad(vca))*dx + (Farad / R / Temp * z_ca*ca)*inner(grad(psi), grad(vca))*dx)
Fpsi = inner(grad(psi), grad(vpsi))*dx - (Farad/(eps0*epsR))*(z_an*an + z_ca*ca + z_fc*fc_function)*vpsi*dx

F = Fpsi + Fan + Fca
J = derivative(F, u)
problem = NonlinearVariationalProblem(F, u, bcs = bcs, J = J)
solver = NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["linear_solver"] = "mumps"

solver.solve()

#this is how you save
###############################################################################
an, ca, psi = u.split(deepcopy=True)

Hdf = HDF5File(mesh.mpi_comm(), "Field_solution.h5", "w")
Hdf.write(mesh, "mesh")
Hdf.write(u, "solution_full")
Hdf.close()
###############################################################################

y = np.linspace(0, 0.015, 1000)
ca_line = np.array([ca(point) for point in y])
an_line = np.array([an(point) for point in y])
psi_line = np.array([psi(point) for point in y])

plt.plot(y, ca_line, color = 'k', linestyle = '-', linewidth = 1.5)
plt.plot(y, an_line, color = 'k', linestyle = '-.', linewidth = 1.5)
plt.legend(['$c^+$', '$c^-$', '$c_{A^-}$'], markerfirst = False)
plt.xlabel('Length [m]')
plt.ylabel('Concentration [mM]')
plt.xlim(0.0, 0.015)
plt.xticks([0, 0.005, 0.010, 0.015])
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.savefig('concentration.png', dpi=800, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, \
            bbox_inches='tight', pad_inches=0.1, frameon=None)
plt.show()

plt.plot(y, psi_line, color = 'k', linestyle = '-', linewidth = 1.5)
plt.xlabel('Length [m]')
plt.ylabel('Electric potential [mV]')
plt.xlim(0.0, 0.015)
plt.xticks([0, 0.005, 0.010, 0.015])
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.legend(['$\psi$'], markerfirst = False)
plt.savefig('potential.png', dpi=800, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, \
            bbox_inches='tight', pad_inches=0.1, frameon=None)
plt.show()
###############################################################################

aftersolveT = timer() 
totime = aftersolveT - startime
print("Number of DOFs: {}".format(ME.dim()))
print("Total time for Simulation : " + str(round(totime)))