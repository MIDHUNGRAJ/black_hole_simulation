import numpy as np

# Constants
G = 1.0  # Gravitational constant (normalized)
M = 1.0  # Mass of black hole (normalized)
c = 1.0  # Speed of light (normalized)
Rs = 2 * G * M / c**2  # Schwarzschild radius

def christoffel_symbols(r):
    """
    Calculate non-zero Christoffel symbols for Schwarzschild metric at radius r.
    Metric signature: (-1, 1, 1, 1) locally ? Standard Schwarzschild coords.
    coords: (t, r, theta, phi)
    
    ds^2 = -(1-Rs/r)dt^2 + (1-Rs/r)^-1 dr^2 + r^2 dtheta^2 + r^2 sin^2theta dphi^2
    """
    # Precompute terms
    fac = 1.0 - Rs / r
    
    # Initialize all to zero
    # indices: 0:t, 1:r, 2:theta, 3:phi
    # We really only need the equations of motion for r, theta, phi for photons
    # usually we can restrict to equatorial plane (theta = pi/2) for symmetry if generic,
    # but for full 3D rendering we need full 3D geodesics or simple rotation logic.
    # Let's do full 4D (or 3 spatial + time parameter) integration for generality.
    
    # Actually, calculating symbols on the fly for every point is slow in python.
    # explicit equations of motion are better.
    pass

def get_derivatives(state):
    """
    Compute derivatives for the geodesic equation.
    State vector: [t, r, theta, phi, ut, ur, utheta, uphi]
    Each component is a numpy array of shape (N_pixels,).
    """
    t, r, theta, phi, ut, ur, utheta, uphi = state
    
    # Schwarzschild factor
    # Add epsilon to prevent div by zero if r hits zero (though we stop at Rs)
    safe_r = np.maximum(r, Rs * 1.0001) 
    S = 1.0 - Rs / safe_r
    
    # Pre-calc terms
    r2 = safe_r**2
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    # Avoid division by zero at poles
    sin_theta = np.where(np.abs(sin_theta) < 1e-6, 1e-6, sin_theta)
    sin2_theta = sin_theta**2
    
    # Christoffel symbols
    Gamma_t_tr = Rs / (2 * safe_r * (safe_r - Rs))
    Gamma_r_tt = Rs * S / (2 * r2)
    Gamma_r_rr = -Rs / (2 * safe_r * (safe_r - Rs))
    Gamma_r_thth = -safe_r * S
    Gamma_r_phph = -safe_r * S * sin2_theta
    Gamma_th_rth = 1.0 / safe_r
    Gamma_th_phph = -sin_theta * cos_theta
    Gamma_ph_rph = 1.0 / safe_r
    Gamma_ph_thph = cos_theta / sin_theta 

    # Derivatives
    dt_dl = ut
    dr_dl = ur
    dtheta_dl = utheta
    dphi_dl = uphi
    
    dut_dl = -2 * Gamma_t_tr * ut * ur
    dur_dl = -(Gamma_r_tt * ut**2 + Gamma_r_rr * ur**2 + Gamma_r_thth * utheta**2 + Gamma_r_phph * uphi**2)
    dutheta_dl = -(2 * Gamma_th_rth * ur * utheta + Gamma_th_phph * uphi**2)
    duphi_dl = -(2 * Gamma_ph_rph * ur * uphi + 2 * Gamma_ph_thph * utheta * uphi)
    
    return np.array([dt_dl, dr_dl, dtheta_dl, dphi_dl, dut_dl, dur_dl, dutheta_dl, duphi_dl])

def integrate_geodesic_batch(y0, steps=200, dl=0.5):
    """
    RK4 integrator for a batch of geodesics.
    y0: initial state array of shape (8, N_pixels)
    Returns:
        final_state: shape (8, N_pixels)
        escaped_mask: boolean array (True if escaped to infinity)
        horizon_mask: boolean array (True if hit horizon)
    """
    y = np.copy(y0)
    
    # Track which rays are done
    mask_active = np.ones(y.shape[1], dtype=bool)
    
    # Result buffers
    final_states = np.copy(y)
    
    for i in range(steps):
        if not np.any(mask_active):
            break
            
        # Extract active rays
        y_active = y[:, mask_active]
        
        k1 = dl * get_derivatives(y_active)
        k2 = dl * get_derivatives(y_active + 0.5 * k1)
        k3 = dl * get_derivatives(y_active + 0.5 * k2)
        k4 = dl * get_derivatives(y_active + k3)
        
        dy = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # Update active
        y[:, mask_active] += dy
        
        # Check conditions on updated y
        r = y[1, :]
        
        # 1. Hit Horizon (r < Rs)
        hit_horizon = (r < Rs * 1.01) & mask_active
        # For those who hit, we stop them there
        final_states[:, hit_horizon] = y[:, hit_horizon]
        mask_active[hit_horizon] = False
        
        # 2. Escaped (r > 300) - Assuming camera starts < 300 or we care about local lensing
        # Actually camera is at say 20. If it goes to > 50 it's likely background.
        escaped = (r > 50) & mask_active 
        final_states[:, escaped] = y[:, escaped]
        mask_active[escaped] = False
    
    # Whatever is left is considered 'active' but we stopped integrating
    final_states[:, mask_active] = y[:, mask_active]
    
    # Classify results
    r_final = final_states[1, :]
    horizon_mask = r_final < Rs * 1.05
    escaped_mask = r_final > Rs * 1.05 # effectively everything else
    
    return final_states, horizon_mask, escaped_mask
