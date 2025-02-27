from __future__ import print_function
from dolfin import (File, Constant, near, IntervalMesh, interval, split, MeshFunction, cells, refine, CompiledSubDomain, DirichletBC, Point, FiniteElement, \
                    MixedElement, FunctionSpace, Function, UserExpression, TestFunctions, derivative, NonlinearVariationalProblem, assign, dot, ds, \
                    inner, grad, dx, SubMesh, solve, plot, TestFunction, TrialFunction, VectorFunctionSpace, as_vector, VectorElement, project, FacetNormal, \
                    interpolate, Expression, NonlinearVariationalSolver, nabla_grad, TrialFunctions, assemble, LinearVariationalSolver, RectangleMesh, \
                    LinearVariationalProblem)
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

################################## mesh part ##################################

mesh = RectangleMesh(Point(0.0, 0.0), Point(15.0E-3, 15.0E-3), 150, 150)
plot(mesh, title = "Original Mesh")
plt.show()

refinement_cycles = 5
for _ in range(refinement_cycles):
    refine_cells = MeshFunction("bool", mesh, 2)
    refine_cells.set_all(False)
    for cell in cells(mesh):
        mp = cell.midpoint()
        if mp.x() > 4.8e-3 and mp.x() < 10.2e-3:
            if abs(mp.y() - 10.0e-3) < 0.2e-3:
                refine_cells[cell] = True
            elif abs(mp.y() - 5.0e-3) < 0.2e-3:
                refine_cells[cell] = True               
        if mp.y() > 4.8e-3 and mp.y() < 10.2e-3:
            if abs(mp.x() - 10.0e-3) < 0.2E-3:
                refine_cells[cell] = True
            elif abs(mp.x() - 5.0e-3) < 0.2e-3:
                refine_cells[cell] = True
    plot(refine_cells, title = "Refined Cells")
    plt.show()    
        
    mesh = refine(mesh, refine_cells)
    plot(mesh, title = "Refined Mesh")
    plt.show()
    
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0)
def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], 15.0e-3)
def bottom_boundary(x, on_boundary):
    return on_boundary and near(x[1], 0.0)
def top_boundary(x, on_boundary):
    return on_boundary and near(x[1], 15.0e-3) 

################################### FE part ###################################   
    
P1 = FiniteElement('P', 'triangle', 2)
element = MixedElement([P1, P1, P1])
ME = FunctionSpace(mesh, element)

subdomain = CompiledSubDomain("x[0] >= 5.0e-3 && x[0] <= 10.0e-3 && x[1] >= 5.0e-3 && x[1] <= 10.0e-3")
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)        
subdomain.mark(subdomains, 1)

fc = Constant(2.0) # FCD
V0_r = FunctionSpace(mesh, 'DG', 0)
fc_function = Function(V0_r)
fc_val = [0.0, fc]

help = np.asarray(subdomains.array(), dtype = np.int32)
fc_function.vector()[:] = np.choose(help, fc_val)
zeroth = plot(fc_function, title = "Fixed charge density, $c^f$")
plt.colorbar(zeroth)
plot(fc_function)
plt.xlim(0.0, 0.015)
plt.xticks([0.0, 0.005, 0.01, 0.015])
plt.ylim(0.0, 0.015)
plt.yticks([0.0, 0.005, 0.01, 0.015])
plt.show()

Sol_c = Constant(1.0)
Poten = 100.0e-3
l_bc_an = DirichletBC(ME.sub(0), Sol_c, left_boundary)
r_bc_an = DirichletBC(ME.sub(0), Sol_c, right_boundary)
l_bc_ca = DirichletBC(ME.sub(1), Sol_c, left_boundary)
r_bc_ca = DirichletBC(ME.sub(1), Sol_c, right_boundary)
l_bc_psi = DirichletBC(ME.sub(2), Constant(-Poten), left_boundary)
r_bc_psi = DirichletBC(ME.sub(2), Constant(Poten), right_boundary)
bcs = [l_bc_an, r_bc_an, l_bc_ca, r_bc_ca, l_bc_psi, r_bc_psi]

u = Function(ME)

V = FunctionSpace(mesh, P1)
an_int = interpolate(Expression('x[0] >= 5.0E-3 and x[0] <= 10.0E-3 && x[1] >= 5.0E-3 and x[1] <= 10.0E-3 ? 0.4142 : 1.0', degree = 1), V)
ca_int = interpolate(Expression('x[0] >= 5.0E-3 and x[0] <= 10.0E-3 && x[1] >= 5.0E-3 and x[1] <= 10.0E-3 ? 2.4142 : 1.0', degree = 1), V)
psi_int = interpolate(Expression('x[0] >= 5.0E-3 and x[0] <= 10.0E-3 && x[1] >= 5.0E-3 and x[1] <= 10.0E-3 ? -22.6322E-3 : 0.0', degree = 1), V)
assign(u.sub(0), an_int)
assign(u.sub(1), ca_int)
assign(u.sub(2), psi_int)

an, ca, psi = split(u)
van, vca, vpsi = TestFunctions(ME)

Fan = D_an*(-inner(grad(an), grad(van))*dx - Farad / R / Temp * z_an*an*inner(grad(psi), grad(van))*dx)
Fca = D_ca*(-inner(grad(ca), grad(vca))*dx - Farad / R / Temp * z_ca*ca*inner(grad(psi), grad(vca))*dx)
Fpsi = inner(grad(psi), grad(vpsi))*dx - (Farad/(eps0*epsR))*(z_an*an + z_ca*ca + z_fc*fc_function)*vpsi*dx

F = Fpsi + Fan + Fca

J = derivative(F, u)
problem = NonlinearVariationalProblem(F, u, bcs = bcs, J = J)
solver = NonlinearVariationalSolver(problem)

solver.parameters["newton_solver"]["linear_solver"] = "mumps"
solver.solve()

#########################  Post - Processing  #################################
an, ca, psi = u.split()

aftersolveT = timer() 

first = plot(an, title = "Anion concentration, $c^-$")
plt.colorbar(first)
plt.xlim(0.0, 0.015)
plt.xticks([0.0, 0.005, 0.01, 0.015])
plt.ylim(0.0, 0.015)
plt.yticks([0.0, 0.005, 0.01, 0.015])
plt.show()

second = plot(ca, title = "Cation concentration, $c^+$")
plt.xlim(0.0, 0.015)
plt.xticks([0.0, 0.005, 0.01, 0.015])
plt.ylim(0.0, 0.015)
plt.yticks([0.0, 0.005, 0.01, 0.015])
plt.colorbar(second)
plt.show()

third = plot(psi, title = "Electric potential, $\psi$")
plt.xlim(0.0, 0.015)
plt.xticks([0.0, 0.005, 0.01, 0.015])
plt.ylim(0.0, 0.015)
plt.yticks([0.0, 0.005, 0.01, 0.015])
plt.colorbar(third)
plt.show()

File('an_ES.pvd') << an
File('ca_ES.pvd') << ca
File('psi_ES.pvd') << psi

y_position = 7.5e-3
x_min = 0.0
x_max = 15.0e-3
N_steps = 500
delta_x = (x_max - x_min)/float(N_steps) 

an_vector = []
for i in range(N_steps):
    x_position = float(i * delta_x)
    an_vector.append(an(x_position, y_position))
x_values = [i * delta_x for i in range(N_steps)]
plt.plot(x_values, an_vector, color = 'k', linestyle = '-', linewidth = 1.0)
plt.title("Anion concentration")
plt.xlim(0.0, 0.015)
plt.xticks([0.0, 0.005, 0.01, 0.015])
plt.grid(color = 'k', linestyle = '-', linewidth = 0.1)
plt.legend(['$c^-$'], loc='best', markerfirst = False, prop={'size': 15})
plt.savefig('an_ES_1D.png', dpi=800, facecolor='w', edgecolor='w', orientation='portrait', transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.show()

ca_vector = []
for i in range(N_steps):
    x_position = float(i * delta_x)
    ca_vector.append(ca(x_position, y_position))
x_values = [i * delta_x for i in range(N_steps)]
plt.plot(x_values, ca_vector, color = 'k', linestyle = '-', linewidth = 1.0)
plt.xlim(0.0, 0.015)
plt.xticks([0.0, 0.005, 0.01, 0.015])
plt.title("Cation concentration")
plt.legend(['$c^+$'], loc='best', markerfirst = False, prop={'size': 15})
plt.grid(color = 'k', linestyle = '-', linewidth = 0.1)
plt.savefig('ca_ES_1D.png', dpi=800, facecolor='w', edgecolor='w', orientation='portrait', transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.show()

psi_vector = []
for i in range(N_steps):
    x_position = float(i * delta_x)
    psi_vector.append(psi(x_position, y_position))
x_values = [i * delta_x for i in range(N_steps)]
plt.plot(x_values, psi_vector, color = 'k', linestyle = '-', linewidth = 1.0)
plt.xlim(0.0, 0.015)
plt.xticks([0.0, 0.005, 0.01, 0.015])
plt.title("Electric potential")
plt.grid(color = 'k', linestyle = '-', linewidth = 0.1)
plt.legend(['$\psi$'], loc='best', markerfirst = False, prop={'size': 15})
plt.savefig('psi_ES_1D.png', dpi=800, facecolor='w', edgecolor='w', orientation='portrait', transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.show()

print("Max. anion concentratrion: {0:1.3f}" .format(max(an_vector)))
print("Min. anion concentratrion: {0:1.3f}" .format(min(an_vector)))
print("Max. cation concentration: {0:1.3f}" .format(max(ca_vector)))
print("Min. cation concentration: {0:1.3f}" .format(min(ca_vector)))
print("Max. potential: {0:1.3f}" .format(max(psi_vector)))
print("Min. potential: {0:1.3f}" .format(min(psi_vector)))
totime = aftersolveT - startime
print("Number of DOFs: {}".format(ME.dim()))
print("Total time for Simulation : " + str(round(totime)))