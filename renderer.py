import numpy as np
import physics
from PIL import Image

def generate_background(width, height):
    """
    Generate a simple procedural star field.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Background color (deep space blue-black)
    img[:, :] = [5, 5, 10]
    
    # Random stars
    num_stars = 2000
    for _ in range(num_stars):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        brightness = np.random.randint(150, 255)
        # Random tint
        r_tint = np.random.randint(200, 255)
        b_tint = np.random.randint(200, 255)
        img[y, x] = [r_tint, brightness, b_tint]
        
    return img

def get_star_color_from_angle(theta, phi):
    """
    Procedural 360 background based on angles.
    theta: [0, pi]
    phi: [0, 2pi] (or any range really)
    """
    # Normalize phi
    phi = phi % (2 * np.pi)
    
    # Milky way band approximation (sinusoidal band)
    band_center = np.pi/2 + 0.5 * np.sin(phi)
    dist_from_band = np.abs(theta - band_center)
    
    intensity = np.exp(-dist_from_band * 5) # Sharply peaked
    
    color = np.array([intensity*20, intensity*30, intensity*50]) # Blueish nebula
    
    # Add noise/stars (hash functionish)
    # Simple pseudo-random static based on angles
    val = np.sin(phi*50) * np.cos(theta*60)
    is_star = val > 0.995
    
    color[is_star] += 200
    
    return np.clip(color, 0, 255).astype(np.uint8)

def render_scene(width, height, fov_deg=60.0):
    print(f"Setting up camera: {width}x{height} FOV={fov_deg}")
    
    # Camera params
    r_cam = 15.0 # Distance in units of M (Rs = 2M = 2.0 with G=c=M=1)
    theta_cam = np.pi / 2.0
    phi_cam = 0.0
    
    # Image plane setup
    aspect = width / height
    fov_rad = np.deg2rad(fov_deg)
    scale = np.tan(fov_rad / 2.0)
    
    # Grid of x, y pixel coordinates [-1, 1]
    x = np.linspace(-1, 1, width) * aspect * scale
    y = np.linspace(1, -1, height) * scale # Flip y so + is up
    xv, yv = np.meshgrid(x, y)
    
    xv = xv.flatten()
    yv = yv.flatten()
    
    N = len(xv)
    print(f"Total rays: {N}")
    
    # Local orthonormal frame at camera
    # e_r (radial outward), e_th (south), e_ph (east)
    # Camera looking at black hole (inward -e_r)
    # Up vector is North (-e_th)
    # Right vector is East (e_ph)
    
    # Directions in local frame
    # Ray direction D = Forward + x*Right + y*Up
    # D = (-1) * e_r + (xv) * e_ph + (yv) * (-e_th)
    # Normalize D
    
    # Components in local orthonormal basis (r, th, ph)
    # k_local_r = -1
    # k_local_th = -yv
    # k_local_ph = xv
    
    norm = np.sqrt((-1)**2 + xv**2 + yv**2)
    k_local_r = -1 / norm
    k_local_th = -yv / norm
    k_local_ph = xv / norm
    k_local_t = 1.0 # Light ray, energy normalized locally
    
    # Convert to coordinate basis vectors
    # Transformation from Orthonormal (hat) to Coordinate basis (no hat)
    # e_hat_t = (1-Rs/r)^-1/2 dt
    # e_hat_r = (1-Rs/r)^1/2 dr
    # e_hat_th = r^-1 dtheta
    # e_hat_ph = (r sin theta)^-1 dphi
    
    # So vectors:
    # U^mu = component * vector
    # Coordinate components u^mu correspond to d/dx^mu
    # u^t = k_local_t / (1-Rs/r)^1/2
    # u^r = k_local_r * (1-Rs/r)^1/2
    # u^th = k_local_th / r
    # u^ph = k_local_ph / (r * sin(theta))
    
    fac = np.sqrt(1.0 - physics.Rs / r_cam)
    
    ut = np.full(N, 1.0 / fac)
    ur = k_local_r * fac
    uth = k_local_th / r_cam
    uph = k_local_ph / (r_cam * np.sin(theta_cam))
    
    # Initial state
    t0 = np.zeros(N)
    r0 = np.full(N, r_cam)
    th0 = np.full(N, theta_cam)
    ph0 = np.full(N, phi_cam)
    
    y0 = np.array([t0, r0, th0, ph0, ut, ur, uth, uph])
    
    print("Tracing rays...")
    final_state, mask_horizon, mask_escaped = physics.integrate_geodesic_batch(y0, steps=400, dl=0.5)
    
    # Construct image
    print("Constructing image...")
    img_data = np.zeros((N, 3), dtype=np.uint8)
    
    # 1. Horizon pixels -> Black
    img_data[mask_horizon] = [0, 0, 0]
    
    # 2. Escaped pixels -> Background
    # Get final angles
    th_f = final_state[2, mask_escaped]
    ph_f = final_state[3, mask_escaped]
    
    # Simple procedural background mapping
    # Note: Vectorized background generation would be better but simple loop for now or map
    # Let's vectorize the star function slightly
    
    # Vectorized background
    phi_norm = ph_f % (2 * np.pi)
    band_center = np.pi/2 + 0.3 * np.sin(phi_norm) + 0.2*np.cos(2*phi_norm) # Wavy
    dist = np.abs(th_f - band_center)
    intensity = np.exp(-dist * 4) * 0.8
    
    # Colors
    R = intensity * 150 + 20
    G = intensity * 100 + 20
    B = intensity * 255 + 50
    
    # Stars
    # Hash for stars
    # We need deterministic noise based on angles
    # Use simple high freq sin
    scale = 100.0
    noise_val = np.sin(ph_f * scale) * np.sin(th_f * scale * 1.05) * np.sin((ph_f + th_f)*scale*0.5)
    is_star = noise_val > 0.99
    
    R[is_star] = 255
    G[is_star] = 255
    B[is_star] = 255

    colors = np.stack([R, G, B], axis=1)
    img_data[mask_escaped] = np.clip(colors, 0, 255).astype(np.uint8)
    
    # Reshape
    img_final = img_data.reshape((height, width, 3))
    
    return Image.fromarray(img_final)

if __name__ == "__main__":
    img = render_scene(320, 240)
    img.save("test_render.png")
    print("Saved test_render.png")
