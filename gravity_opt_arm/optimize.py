import math
from typing import Tuple, Optional, List, Dict
from core import fk_3r, potential_energy                  
from ik2r import wrist_point, solve_2R_to_point, assemble_3r

def search_min_energy_to_point(
    target_xy: Tuple[float, float],
    L: Tuple[float, float, float],
    M: Tuple[float, float, float],
    g: float,
    phi_fixed: Optional[float],     
    phi_step_deg: float,             
    refine_deg: float,               
    do_refine: bool,               
    joint_limits: Optional[List[Tuple[float, float]]] ,  
) -> Dict[str, object]:

    def within_limits(theta):
        
        if joint_limits is None:
            return True
        for ang, (lo, hi) in zip(theta, joint_limits):
            if ang < lo - 1e-9 or ang > hi + 1e-9:
                return False
        return True

    def scan(phi_values):
        
        best = {"V": float("inf"), "phi": None, "theta": None, "branch": None}
        for phi in phi_values:

            xw, yw = wrist_point(target_xy, phi, L[2])
            
            sols = solve_2R_to_point(xw, yw, L[0], L[1])
            
            if not sols:
                continue

            for idx, (t1, t2) in enumerate(sols):
                
                th = assemble_3r(phi, t1, t2)
                if not within_limits(th):
                    continue
                
                x, y = fk_3r(th, L)
                if (x - target_xy[0])**2 + (y - target_xy[1])**2 > 1e-12:
                    continue
                
                V = potential_energy(th, L, M, g)
                
                if V < best["V"]:
                    best.update({"V": V, "phi": phi, "theta": th, "branch": "down" if idx == 0 else "up"})
        return best

    
    if phi_fixed is None:
        step = math.radians(phi_step_deg)
        N = int((2*math.pi)/step) + 1
        phi_vals = [(-math.pi + i*step) for i in range(N)]
    else:
        phi_vals = [phi_fixed]   

    best = scan(phi_vals)       

    
    if do_refine and best["phi"] is not None and phi_fixed is None:
        center = best["phi"]
        delta = math.radians(5.0)    
        step = math.radians(refine_deg)
        phi_ref = [center + j*step for j in range(int(-delta/step), int(delta/step)+1)]
        best_ref = scan(phi_ref)    
        if best_ref["phi"] is not None and best_ref["V"] < best["V"]:
            best = best_ref         

    if best["phi"] is None:
       
        return {"ok": False, "reason": "Target not reachable."}

    
    th = best["theta"]
    return {
        "ok": True,
        "theta_rad": th,
        "theta_deg": tuple(math.degrees(t) for t in th),
        "phi_rad": best["phi"],
        "phi_deg": math.degrees(best["phi"]),
        "branch": best["branch"],
        "V_J": best["V"],
    }
