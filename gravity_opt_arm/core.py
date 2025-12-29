
import math                       
from typing import Tuple, Sequence 

def fk_3r(theta: Sequence[float], L: Sequence[float]) -> Tuple[float, float]:
    """Forward kinematics: (θ1,θ2,θ3) -> (x,y) for planar 3R"""

    t1, t2, t3 = theta         
    L1, L2, L3 = L               

    x = (L1*math.cos(t1)
         + L2*math.cos(t1+t2)
         + L3*math.cos(t1+t2+t3))
    y = (L1*math.sin(t1)
         + L2*math.sin(t1+t2)
         + L3*math.sin(t1+t2+t3))
    
    return x, y                  

def com_y(theta: Sequence[float], L: Sequence[float]) -> Tuple[float, float, float]:
    """Heights of COMs (mid-link assumption)"""

    t1, t2, t3 = theta          
    L1, L2, L3 = L               

    y1 = 0.5*L1*math.sin(t1)

    y2 = L1*math.sin(t1) + 0.5*L2*math.sin(t1+t2)

    y3 = (L1*math.sin(t1)
          + L2*math.sin(t1+t2)
          + 0.5*L3*math.sin(t1+t2+t3))
    
    return y1, y2, y3

def potential_energy(theta: Sequence[float], L: Sequence[float],
                     M: Sequence[float], g: float = 9.81) -> float:
    
    """V(θ) = g Σ m_i y_COM_i(θ)"""
    y1, y2, y3 = com_y(theta, L) 
    m1, m2, m3 = M               
    
    return g * (m1*y1 + m2*y2 + m3*y3)
