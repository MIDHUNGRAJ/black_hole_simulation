import argparse
import sys
import time
from renderer import render_scene

def main():
    parser = argparse.ArgumentParser(description="Schwarzschild Black Hole Ray Tracer")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--output", type=str, default="blackhole.png", help="Output filename")
    parser.add_argument("--fov", type=float, default=60.0, help="Field of View")
    
    args = parser.parse_args()
    
    print(f"Starting render: {args.width}x{args.height}")
    start_time = time.time()
    
    try:
        img = render_scene(args.width, args.height, fov_deg=args.fov)
        img.save(args.output)
        elapsed = time.time() - start_time
        print(f"Render complete! Saved to {args.output}")
        print(f"Time taken: {elapsed:.2f} seconds")
    except ImportError as e:
        print("Error: Missing dependencies.")
        print("Please ensure numpy and pillow are installed:")
        print("pip install numpy matplotlib pillow")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
