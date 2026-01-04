import math
from typing import Tuple, Sequence


def wrap_to_pi(a: float) -> float:

    return (a + math.pi) % (2*math.pi) - math.pi


def wrist_point(target_xy: Tuple[float, float], phi: float, L3: float) -> Tuple[float, float]:
    """(xw,yw) = target - L3 * [cos(phi), sin(phi)]"""

    x_star, y_star = target_xy               
    
    return (x_star - L3*math.cos(phi), y_star - L3*math.sin(phi))


def solve_2R_to_point(xw: float, yw: float, L1: float, L2: float) -> Tuple[Tuple[float,float], ...]:
    """Compute up to two 2R IK solutions (θ1, θ2) for a given wrist point."""

    d2 = xw*xw + yw*yw                  
    d = math.sqrt(d2)                   

    if d > L1 + L2 + 1e-12 or d < abs(L1 - L2) - 1e-12:
        return tuple()                  

    
    c2 = (d2 - L1*L1 - L2*L2) / (2.0*L1*L2)
    c2 = max(-1.0, min(1.0, c2))       
    t2a = math.acos(c2)                
    t2b = -t2a                         

    sols = []
    for t2 in (t2a, t2b):
        
        k1 = L1 + L2*math.cos(t2)
        k2 = L2*math.sin(t2)
        
        t1 = math.atan2(yw, xw) - math.atan2(k2, k1)
        sols.append((wrap_to_pi(t1), wrap_to_pi(t2)))

    
    if abs(t2a - t2b) < 1e-12:
        sols = sols[:1]
    return tuple(sols)


def assemble_3r(phi: float, theta1: float, theta2: float) -> Tuple[float, float, float]:
    """θ3 = φ - θ1 - θ2 (wrap)"""
    t3 = wrap_to_pi(phi - theta1 - theta2) 
    return (wrap_to_pi(theta1), wrap_to_pi(theta2), t3)
