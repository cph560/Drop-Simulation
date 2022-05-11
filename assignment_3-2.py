import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from assignment_3_helper import LCPSolve, assignment_3_render

# DEFINE GLOBAL PARAMETERS
L = 0.4
MU = 0.3
EP = 0.5
dt = 0.01
m = 0.3
g = np.array([0., -9.81, 0.])
rg = 1./12. * (2 * L * L)
M = np.array([[m, 0, 0], [0, m, 0], [0, 0, m * rg]])
Mi = np.array([[1./m, 0, 0], [0, 1./m, 0], [0, 0, 1./(m * rg)]])
DELTA = 0.001
T = 150


def get_contacts(q):
    """
        Return jacobian of the lowest corner of the square and distance to contact
        :param q: <np.array> current configuration of the object
        :return: <np.array>, <float> jacobian and distance
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    q=q.reshape((3,1))
    l=0.4

    R=np.squeeze(np.array([[np.cos(q[2]),-np.sin(q[2])],[np.sin(q[2]),np.cos(q[2])]]))
    t=np.array([q[0],q[1]])

    qo=np.array([[-l/2, l/2,l/2,-l/2],[-l/2,-l/2,l/2,l/2]])
    qc=np.matmul(R,qo)+t

    phi=qc[1,:].min()

    min_xy=np.argmin(qc[1,],axis=0)

    min_corner=np.array([qc[0,min_xy],qc[1,min_xy]]).reshape((2,1))
    r=min_corner-t.reshape((2,1))
    n = np.array([[0], [1]])
    jac = np.array([[n[1, 0], n[0, 0]], [-n[0, 0], n[1, 0]],
                    [-r[0, 0] * n[0, 0] -r[1, 0] * n[1, 0], r[0, 0] * n[1, 0] - r[1, 0] * n[0, 0]]])

    # ------------------------------------------------
    return jac, phi


def form_lcp(jac, v):
    """
        Return LCP matrix and vector for the contact
        :param jac: <np.array> jacobian of the contact point
        :param v: <np.array> velocity of the center of mass
        :return: <np.array>, <np.array> V and p
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    Jn = jac[:,1]
    Jt = jac[:,0]
    Jnt=np.transpose(Jn)
    Jtt=np.transpose(Jt)
    Jhat=np.zeros((3,3))


    Jhat[:,0]+=Jn
    Jhat[:, 1] += (-Jt)
    Jhat[:, 2] += Jt



    S=np.matmul(np.matmul(np.transpose(Jhat),Mi),Jhat)*dt

    Vs=np.array([[0,0,0,0],[0,0,0,1],[0,0,0,1],[MU,-1,-1,0]])
    Vs[0:3, 0:3] += S

    fe=np.matmul(M,np.transpose(g)).reshape((3, 1))
    v = v.flatten()

    p1=np.matmul(Jnt,((1+EP)*v+dt*(np.matmul(Mi,fe).flatten())))
    p2=np.matmul(-Jtt,(v+dt*(np.matmul(Mi,fe).flatten())))
    p3=np.matmul(Jtt,(v+dt*(np.matmul(Mi,fe).flatten())))


    V = Vs  # TODO: Replace None with your result


    p = np.array([p1,p2,p3,0])
    print(p)



    # ------------------------------------------------
    return V, p


def step(q, v):
    """
        predict next config and velocity given the current values
        :param q: <np.array> current configuration of the object
        :param v: <np.array> current velocity of the object
        :return: <np.array>, <np.array> q_next and v_next
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    jac, phi = get_contacts(q)
    fe=np.matmul(M,np.transpose(g)).reshape((3, 1))

    v = v.reshape((3, 1))
    q=q.reshape((3, 1))


    if phi>DELTA:

        vt1=v+dt*np.matmul(Mi,fe)
        qt1=q+dt*vt1

    else:
        V,p=form_lcp(jac,v)
        fc=lcp_solve(V, p)
        Jnfn = jac[:, 1] * fc[0]
        Jtft1 = jac[:, 0] * fc[1]
        Jtft2 = jac[:, 0] * fc[2]
        qp=np.array([0,DELTA,0]).reshape((3,1))
        vt1=v+dt*np.matmul(Mi,(fe+Jnfn.reshape((3,1))-Jtft1.reshape((3,1))+Jtft2.reshape((3,1))))

        qt1=q+dt*vt1+qp


    q_next = qt1  # TODO: Replace None with your result
    v_next = vt1

    # ------------------------------------------------
    return q_next, v_next


def simulate(q0, v0):
    """
        predict next config and velocity given the current values
        :param q0: <np.array> initial configuration of the object
        :param v0: <np.array> initial velocity of the object
        :return: <np.array>, <np.array> q and v trajectory of the object
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    qt,vt=step(q0, v0)

    qs=np.zeros((3, T))
    vs=np.zeros((3, T))
    qs[:,0] +=q0

    qs[:, 1] += qt.flatten()
    vs[:,0] +=v0
    vs[:, 1] += vt.flatten()

    for i in range(2,T):
        qt,vt=step(qt,vt)
        qs[:, i] += qt.flatten()
        vs[:, i] += vt.flatten()


    q = qs  # TODO: Replace with your result


    v = vs

    # ------------------------------------------------
    return q, v


def lcp_solve(V, p):
    """
        DO NOT CHANGE -- solves the LCP
        :param V: <np.array> matrix of the LCP
        :param p: <np.array> vector of the LCP
        :return: renders the trajectory
    """
    sol = LCPSolve(V, p)
    f_r = sol[1][:3]

    return f_r


def render(q):
    """
        DO NOT CHANGE -- renders the trajectory
        :param q: <np.array> configuration trajectory
        :return: renders the trajectory
    """
    assignment_3_render(q)


if __name__ == "__main__":
    # to test your final code, use the following initial configs
    q0 = np.array([0.0, 1.5, np.pi / 180. * 30.])
    v0 = np.array([0., -0.2, 0.])
    q, v = simulate(q0, v0)

    plt.plot(q[1, :])
    

    render(q)
