import numpy as np
import matplotlib.pyplot as plt
from pyproj import Geod
from shapely.geometry import Polygon, Point
from skyfield.api import wgs84


class HexagonGrid:
    """
    A class for creating and visualizing hexagonal grids on geographic coordinates.
    """
    
    def __init__(self, center_lat, center_lon, radius_km):
        """
        Initialize a hexagonal grid with a center point and radius.
        
        Args:
            center_lat (float): Latitude of the center point
            center_lon (float): Longitude of the center point
            radius_km (float): Radius of the hexagon in kilometers
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_km = radius_km
        self.vertices = self._generate_vertices()
        self.geod = Geod(ellps="WGS84")
        
    def _generate_vertices(self):
        """Generate vertices for the hexagonal cell centered at (center_lat, center_lon) with given radius."""
        geod = Geod(ellps="WGS84")
        vertices = []
        for angle in range(0, 360, 60):  # 6 vertices, spaced by 60 degrees
            lon, lat, _ = geod.fwd(self.center_lon, self.center_lat, angle, self.radius_km * 1000)
            vertices.append((lat, lon))
        return vertices
    
    def get_vertices(self):
        """Return the vertices of the hexagon."""
        return self.vertices
    
    def convert_vertices_to_wgs84(self, vertices):
        arr = np.array(vertices)
        zzz_lats = arr[:, 0]
        zzz_lons = arr[:, 1]
        zzzazzar = wgs84.latlon(zzz_lats, zzz_lons)
        return zzzazzar
    
    def sort_vertices_by_latitude(self):
        """Sort vertices by latitude in descending order."""
        return sorted(self.vertices, key=lambda x: x[0], reverse=True)
    
    def sort_vertices_counterclockwise(self):
        """Sort vertices in counterclockwise order around the center."""
        # Calculate centroid
        center_lat = sum(v[0] for v in self.vertices) / len(self.vertices)
        center_lon = sum(v[1] for v in self.vertices) / len(self.vertices)
        
        # Function to calculate angle between point and horizontal axis
        def get_angle(point):
            return np.arctan2(point[0] - center_lat, point[1] - center_lon)
        
        # Sort vertices by angle
        sorted_vertices = sorted(self.vertices, key=get_angle, reverse=True)
        
        # Add first vertex at the end to close the polygon
        return sorted_vertices + [sorted_vertices[0]]
    
    def get_north_edge(self):
        """Get the two points of the northern edge of the hexagon."""
        vertices_sorted = self.sort_vertices_by_latitude()
        # The two highest latitude points define the northern edge
        northern_point1 = vertices_sorted[0]
        northern_point2 = vertices_sorted[1]
        return northern_point1, northern_point2
    
    def calculate_midpoint(self, lat1, lon1, lat2, lon2):
        """Calculate the midpoint between two geographic points."""
        # Compute midpoint
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        return mid_lat, mid_lon
    
    def generate_grid_points(self, spacing_km=None, num_points=None):
        """
        Generate grid points inside the hexagonal cell.
        
        Args:
            spacing_km (float, optional): Spacing between grid points in kilometers
            num_points (int, optional): Target number of points to generate (approximate)
            
        Returns:
            list: List of (lat, lon) tuples representing grid points
            
        Note:
            - If spacing_km is provided, points will be generated with that fixed spacing
            - If num_points is provided, spacing will be calculated to generate approximately that many points
            - If both are provided, spacing_km takes precedence
            - If neither is provided, defaults to a spacing that would generate approximately 100 points
        """
        # Convert hexagon vertices to a shapely Polygon
        hexagon = Polygon(self.vertices)
        
        # Find bounding box for hexagon
        min_lat, min_lon, max_lat, max_lon = hexagon.bounds
        
        # Calculate the approximate area of the hexagon in square degrees
        # (This is an approximation that works reasonably well for small areas)
        width_deg = max_lon - min_lon
        height_deg = max_lat - min_lat
        area_deg2 = width_deg * height_deg * 0.866  # 0.866 is an approximation for hexagon/rectangle area ratio
        
        # Determine spacing to use
        if spacing_km is not None:
            # Use the provided spacing
            spacing_deg = spacing_km / 111  # Approximate degree spacing (1 degree ~ 111 km at equator)
        elif num_points is not None:
            # Calculate spacing to achieve the target number of points
            points_per_deg2 = num_points / area_deg2
            spacing_deg = np.sqrt(1 / points_per_deg2)
        else:
            # Default to approximately 100 points
            points_per_deg2 = 100 / area_deg2
            spacing_deg = np.sqrt(1 / points_per_deg2)
        
        # Create grid points over the bounding box
        grid_points = []
        
        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                point = Point(lat, lon)
                # Check if the point is inside the hexagon
                if hexagon.contains(point):
                    grid_points.append((lat, lon))
                lon += spacing_deg
            lat += spacing_deg
        
        return grid_points
    
    def plot_hexagon(self, show=True):
        """
        Plot only the hexagonal cell and its center.
        
        Args:
            show (bool): Whether to display the plot immediately
        """
        # Close the hexagon by adding the first vertex at the end
        vertices_closed = self.vertices + [self.vertices[0]]
        lats, lons = zip(*vertices_closed)
        
        plt.figure(figsize=(8, 8))
        plt.plot(lons, lats, marker='o', label="Cell Edges")
        plt.scatter(self.center_lon, self.center_lat, color='red', label="Cell Center")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Hexagonal Cell (Radius: {self.radius_km} km)")
        plt.legend()
        plt.grid()
        
        if show:
            plt.show()
        
        return plt.gcf()  # Return the figure object
    
    def plot_with_grid_points(self, grid_points, show=True, title=None):
        """
        Plot the hexagonal cell with grid points.
        
        Args:
            grid_points (list): List of (lat, lon) tuples representing grid points
            show (bool): Whether to display the plot immediately
            title (str, optional): Custom title for the plot
        """
        # Close the hexagon by adding the first vertex at the end
        vertices_closed = self.vertices + [self.vertices[0]]
        lats, lons = zip(*vertices_closed)
        
        # Unzip grid points
        if grid_points:
            grid_lats, grid_lons = zip(*grid_points)
        else:
            grid_lats, grid_lons = [], []
        
        plt.figure(figsize=(8, 8))
        plt.plot(lons, lats, marker='o', label="Cell Edges")
        plt.scatter(self.center_lon, self.center_lat, color='red', s=20, label="Cell Center")
        
        if grid_points:
            plt.scatter(grid_lons, grid_lats, color='blue', s=10, label=f"Grid Points ({len(grid_points)} points)")
        
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Hexagonal Cell with Grid Points (Radius: {self.radius_km} km, {len(grid_points)} points)")
        
        plt.legend()
        plt.grid()
        
        if show:
            plt.show()
        
        return plt.gcf()  # Return the figure object
    
    

    def Point_initial_location_generation(self, cell_center_lat, cell_center_lon, radius):
        

        # Original center
        cell_center = wgs84.latlon(cell_center_lat, cell_center_lon)  # GS-BR
        # Initialize Geod with WGS84 ellipsoid
        geod = Geod(ellps='WGS84')
        # Generate random angle (0 to 2π) and radius (0 to R)
        theta = np.random.uniform(0, 2 * np.pi)
        r = radius * 1000 * np.sqrt(np.random.uniform())  # Ensures uniform distribution
        
        # Compute new latitude and longitude
        lon_new, lat_new, _ = geod.fwd(
            cell_center.longitude.degrees,
            cell_center.latitude.degrees,
            np.degrees(theta),  # Convert angle to degrees
            r)
        HAPS_initial_location = wgs84.latlon(lat_new, lon_new)
        
        return HAPS_initial_location
    

"""
# Example usage:
if __name__ == '__main__':
    # Define center coordinates and radius
    base_lat, base_lon = 53.110987, 8.851239
    radius_km = 25
    
    # Create a hexagonal grid
    hex_grid = HexagonGrid(base_lat, base_lon, radius_km)
    
    # Plot just the hexagon
    hex_grid.plot_hexagon()
    
    # Generate grid points using fixed spacing
    spacing_km = 1  # 1 km spacing between grid points
    grid_points_fixed = hex_grid.generate_grid_points(spacing_km=spacing_km)
    
    # Plot with grid points (fixed spacing)
    hex_grid.plot_with_grid_points(grid_points_fixed, show=False)
    plt.title(f"Hexagonal Cell with Grid Points (Fixed {spacing_km}km spacing)")
    plt.savefig("hexagon_fixed_spacing.png")
    plt.close()
    
    # Generate grid points using approximate number of points
    num_points = 200  # Target number of points
    grid_points_num = hex_grid.generate_grid_points(num_points=num_points)
    
    # Plot with grid points (based on number)
    hex_grid.plot_with_grid_points(grid_points_num, show=False)
    plt.title(f"Hexagonal Cell with Grid Points (Target: ~{num_points} points, Actual: {len(grid_points_num)})")
    plt.show()
    """