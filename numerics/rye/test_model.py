"""Targeted regression tests against exact examples, independent enumeration separately."""
import unittest
import numpy as np
from model import solve
class ModelTests(unittest.TestCase):
    def test_hd_exact(self):
        x=[1,-.5,-.5,-1,-1,-1];u=[dict(E=1,Pc=1,Pd=1,eta=1,delta=0)]
        for r,H,D,want in [(0,3,None,0),(0,None,2,0),(.249,3,2,2),(.25,3,2,0)]:
            a=solve(x,u,r,H=H,D=D,save=False);self.assertEqual(a['status'],want)
    def test_thermal_exact(self):
        x=[1,-1,-.8];u=[dict(E=1,Pc=1,Pd=1,eta=1,delta=.1)]
        for r,want in [(4/95-1e-4,2),(4/95,0)]:
            self.assertEqual(solve(x,u,r,H=1,ens_lim=.9,save=False)['status'],want)
    def test_delta_zero(self):
        a=solve([1,-1,-.8],[dict(E=1,Pc=1,Pd=1,eta=1,delta=0)],0,H=1,ens_lim=.8,save=False)
        self.assertEqual(a['status'],0)
    def test_linear_series(self):
        a=solve([-1,1,-1],[dict(E=0,Pc=0,Pd=0,eta=1,delta=0)],0,D=1,save=False)
        self.assertEqual(a['status'],0);self.assertEqual(a['D_linear'],1);self.assertEqual(a['D_cyclic'],2)
if __name__=='__main__':unittest.main()
