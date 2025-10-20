import MDAnalysis as mda
import numpy as np
from decimal import Decimal, getcontext
import numpy as np
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.analysis import distances
import pandas as pd
from scipy.stats import norm
import time
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
from decimal import Decimal, getcontext
from scipy import constants as cst
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.distances import self_capped_distance
from MDAnalysis.lib.distances import capped_distance
import numpy as np
from collections import defaultdict  # Make sure to import defaultdict
from MDAnalysis.lib.distances import capped_distance

start_time = time.time()

# Read the argument (frame index ) passed from the Bash script 
index = 1#int(sys.argv[1]) 

### Please "DO NOT" change number of segments !
num_segments = 1
#print("The number of the segments is :\n",num_segments)

#C6 = 0.21558       KJ mol^-1 nm^6
#C12 = 0.0023238    KJ mol^-1 nm^12
cut_off = 12             
epsilon=1.19503 # kcal/mol          
sigma=4.7                 
atom_mass= 72
r1=9
rc=12

r = np.arange(0.01,rc,0.001)

def get_abc(alpha,r1,rc):
    A = -((alpha+4)*rc-(alpha+1)*r1)/(rc**(alpha+2)*(rc-r1)**2)
    A = alpha*A
    B = ((alpha+3)*rc-(alpha+1)*r1)/(rc**(alpha+2)*(rc-r1)**3)
    B = alpha*B
    C = (1/rc**alpha) - (A/3)*(rc-r1)**3 - (B/4)*(rc-r1)**4
    return A,B,C

def get_s(r,alpha,r1,rc):
    A,B,C = get_abc(alpha,r1,rc)
    S = A*(r-r1)**2 + B*(r-r1)**3
    return S

def get_switched_force(r,alpha,r1,rc):
    unswitched = alpha/r**(alpha+1)
    switched = unswitched + get_s(r,alpha,r1,rc)
    switched[r<r1] = unswitched[r<r1]
    switched[rc<=r] = 0.0
    switched[r<0.04] = 0.0
    return switched


def F_vdw_switch(r):
    Fs_12= get_switched_force(r,alpha=12,r1=r1,rc=rc)
    Fs_6 = get_switched_force(r,alpha=6,r1=r1,rc=rc)
    A=  4 * epsilon * sigma**12
    C=  4 * epsilon * sigma**6
    sign = +1
    F_vdw_switch = sign * (A*Fs_12 - C*Fs_6)
    return  F_vdw_switch



class CirclePlot:
    def __init__(self, origin_yy, origin_zz, cylinder_radius, RMax,cutoff, num_points=100):
        """
        Initialize the circle with center at (origin_yy, origin_zz) and the specified cylinder_radius.
        Parameters:
        - origin_yy: Y-coordinate of the center.
        - origin_zz: Z-coordinate of the center.
        - cylinder_radius: Radius of the circle.
        - num_points: Number of points used to plot the circle (default is 100).
        """
        self.origin_yy = origin_yy
        self.origin_zz = origin_zz
        self.cylinder_radius = cylinder_radius
        self.RMax = RMax # cylinder_radius+12 Angestrum
        self.cutoff = cutoff
        self.num_points = num_points
        self.generate_circle_points()
        
    def check_position_inside_cutoff(self, y, z):
        """
        This method checks the position of a point (y, z) relative to the circle.
        It returns whether the point is on, above, or below the cut of radius.
        The circle equation is: (y - Origin_YY)^2 + (z - Origin_ZZ)^2 = Radius^2.
        """
        # Calculate the squared distance from the center
        distance_squared = (y - self.origin_yy)**2 + (z - self.origin_zz)**2
        RMax_radius_squared = self.RMax**2
        
        if distance_squared == RMax_radius_squared:
            return "on"
        elif distance_squared > RMax_radius_squared:
            return "above"
        else:
            return "below"
        
            
    def check_position_inside_sharp_cutoff(self, y, z):

        distance_squared = (y - self.origin_yy)**2 + (z - self.origin_zz)**2
        RMax_radius_squared_sharp = self.cylinder_radius**2 + self.cutoff**2
        
        if distance_squared == RMax_radius_squared_sharp:
            return "on"
        elif distance_squared > RMax_radius_squared_sharp:
            return "above"
        else:
            return "below"
        
    def check_position(self, y, z):
        """
        This method checks the position of a point (y, z) relative to the circle.
        It returns whether the point is on, above, or below the cut of radius.
        The circle equation is: (y - Origin_YY)^2 + (z - Origin_ZZ)^2 = Radius^2.
        """
        # Calculate the squared distance from the center
        distance_squared = (y - self.origin_yy)**2 + (z - self.origin_zz)**2
        Cylinder_radius_squared = self.cylinder_radius**2
        
        if distance_squared == Cylinder_radius_squared:
            return "on"
        elif distance_squared > Cylinder_radius_squared:
            return "above"
        else:
            return "below"
        
    def generate_circle_points(self):
        """
        This method generates points that satisfy the circle equation in the y-z plane
        by parameterizing the circle. Returns a list of (y, z) points along the circle.
        """
        self.theta = np.linspace(0, 2 * np.pi, self.num_points)
        self.y_points_circle_cylinder = self.origin_yy + self.cylinder_radius * np.cos(self.theta)
        self.z_points_circle_cylinder = self.origin_zz + self.cylinder_radius * np.sin(self.theta)
        self.y_points_circle_RMax = self.origin_yy + self.RMax * np.cos(self.theta)
        self.z_points_circle_RMax = self.origin_zz + self.RMax * np.sin(self.theta)
        


    def interpolate_point(self, A, B, S):
        """
        Interpolate between points A and B using parameter S (0 <= S <= 1).
        if s=0, the result is vector_1.
        if s=1, the result is vector_2.
        if 0<S<1, the result is between A and B.
        if S>1 or S<0, the result extrapolates beyond A or B.
        """
        vector_1 = (A[1] - self.origin_yy, A[2] - self.origin_zz)  
        vector_2 = (B[1] - self.origin_yy, B[2] - self.origin_zz) 
        vector_diff = (vector_2[0] - vector_1[0], vector_2[1] - vector_1[1])
        scaled_vector = (vector_diff[0] * S, vector_diff[1] * S)
        result_vector = (vector_1[0] + scaled_vector[0], vector_1[1] + scaled_vector[1])
        return result_vector
    
 
        
    def compute_intersection(self, A, B):
        Ay, Az = A[1], A[2]
        By, Bz = B[1], B[2]
        dy, dz = By - Ay, Bz - Az
        a = dy**2 + dz**2
        b = 2 * (dy * (Ay - self.origin_yy) + dz * (Az - self.origin_zz))
        c = (Ay - self.origin_yy)**2 + (Az - self.origin_zz)**2 - self.cylinder_radius**2 
        
        discriminant = b**2 - 4*a*c 
        #print("discriminant:",discriminant)
        intersections = []
        if discriminant < 0:
            #print("No intersection")
            intersections.append(())
            return intersections  # No intersection, return an empty list

        elif discriminant == 0: ###  Tangent case 
            #print("Only one intersection")
            S1 = -b / (2 * a)
            if 0 <= S1 <= 1:
                inter_y = Ay + S1 * dy
                inter_z = Az + S1 * dz
                intersections.append(())  # Store S1 along with the intersection
                #intersections.append((S1, inter_y, inter_z))  # Store S1 along with the intersection
            else:
                intersections.append(())
            return intersections

        elif discriminant > 0:
            #print("Two intersections")
            sqrt_discriminant = np.sqrt(discriminant)
            S1 = (-b + sqrt_discriminant) / (2 * a)
            S2 = (-b - sqrt_discriminant) / (2 * a)

            # Store the intersections with their corresponding S values
            for S in [S1, S2]:
                inter_y = Ay + S * dy
                inter_z = Az + S * dz
                intersections.append((S, inter_y, inter_z))
            return intersections
    
            
    
    def plot_circle(self, A, B, S_1, S_2,DELTA_S,Virial, New_Virial):
        """
        Plot the cylinder radius in the 2D YZ plane, mark the center, and show points A, B, and S.
        Also check and print the position of each point relative to the circle.
        """
        plt.figure(figsize=(6,6))
        plt.plot(self.y_points_circle_cylinder, self.z_points_circle_cylinder, linewidth=0.2)
        plt.plot(self.y_points_circle_RMax, self.z_points_circle_RMax, label="Cutoff")
        plt.scatter([self.origin_yy], [self.origin_zz], color='red')  
        # Plot points A and B 
        plt.scatter([A[1]], [A[2]], color='blue',  label="Point A")   # Point A(y, z)
        plt.scatter([B[1]], [B[2]], color='green')   # Point B(y, z)
        # Check position of points A and B
        #print(f"Point A is {self.check_position(A[1], A[2])} the Cylinder radius.")
        #print(f"Point B is {self.check_position(B[1], B[2])} the Cylinder radius.")
        # Compute intersection points 
        intersections = self.compute_intersection(A, B)
        #print("INT:",intersections)
        count_non_empty_tuples = len([x for x in intersections if x]) 
        #print("# OF intersections:",(count_non_empty_tuples))
        
        if count_non_empty_tuples == 2:
            INT_Y1 = float(intersections[0][1])
            INT_Z1 = float(intersections[0][2])
            INT_Y2 = float(intersections[1][1])
            INT_Z2 = float(intersections[1][2])
            plt.scatter(INT_Y1, INT_Z1, color='purple')
            plt.scatter(INT_Y2, INT_Z2, color='purple')
            plt.title(f"Cylinder Radius in YZ Plane \n" f"S Values: ({S_1:.5f}, {S_2:.5f},{DELTA_S:.5f}, {Virial:.5f},{New_Virial:.5f})")
            filename = f"circle_intersection_{INT_Y1:.2f}_{INT_Z1:.2f}_{INT_Y2:.2f}_{INT_Z2:.2f}_{Virial}_{New_Virial}.png"
            plt.xlabel("Y-axis")
            plt.ylabel("Z-axis")
            plt.legend()
            plt.axis("equal") 
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            #plt.show()

        elif count_non_empty_tuples == 1:
            INT_Y1, INT_Z1 = intersections[0][1], intersections[0][2]
            if np.isnan(INT_Y1) or np.isnan(INT_Z1):
                INT_Y1, INT_Z1 = intersections[1][1], intersections[1][2]  
            else:
                INT_Y1, INT_Z1 = intersections[0][1], intersections[0][2]  

            plt.title(f"Cylinder Radius in YZ Plane \n" f"S Values: ({S_1:.5f}, {S_2:.5f},{DELTA_S:.5f}, {Virial:.5f},{New_Virial:.5f})")
            filename = f"circle_intersection_{INT_Y1:.2f}_{INT_Z1:.2f}_{Virial}_{New_Virial}.png"
            plt.scatter([self.origin_yy], [self.origin_zz], color='red')  
            plt.scatter(INT_Y1, INT_Z1, color='purple')
            plt.xlabel("Y-axis")
            plt.ylabel("Z-axis")
            plt.legend()
            plt.axis("equal") 
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            #plt.show()

        else:
            plt.scatter([self.origin_yy], [self.origin_zz], color='red')
            plt.title(f"Cylinder Radius in YZ Plane \n" f"S Values: ({S_1:.5f}, {S_2:.5f},{DELTA_S:.5f}, {Virial:.5f},{New_Virial:.5f})")
            filename = f"circle_intersection_{Virial}_{New_Virial}.png"
            plt.xlabel("Y-axis")
            plt.ylabel("Z-axis")
            plt.legend()
            plt.axis("equal")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            #plt.show()


class Irving_Kirwood:
    def __init__(self, norm_displacement_vector, Radius, Z, num_segments, X , r_MIC_x):
        self. norm_displacement_vector = norm_displacement_vector
        self.Radius=Radius
        self.Z = Z  # Box size (Length of box in Z direction)
        self.num_segments = num_segments  # Number of segments
        self.X = X  # Base virial (unmodified)
        self.segment_length = float(Z / num_segments)  # Length of each segment
        self.r_MIC_x = r_MIC_x
        
        

    def find_segment(self, z):
        
        #print(z)
        z = z % self.Z 
        #print(z)
        
        for idx, (low, high) in enumerate(self.get_segments()):
            if (low <= z < high) or (idx == self.num_segments - 1 and abs(z - self.Z) < 1e-10):
                return idx, low, high
        return None, None, None  # Return None if out of bounds

    def get_segments(self):
        """Return the list of segments based on the box's dimensions."""
        segments = []
        offset = 0.0000  # Define the shift for the first segment
        for i in range(self.num_segments):
            low = offset + i * self.segment_length
            high = offset + (i + 1) * self.segment_length
            segments.append((low, high))
        return segments
    
        
    def interaction(self, A, B, z_A, z_B):
        output_file = "interaction_log.txt"

      
        # Open file in append mode ('a' keeps previous logs)
        with open(output_file, "a") as file:
            seg_A, low_A, high_A = self.find_segment(z_A)
            seg_B, low_B, high_B = self.find_segment(z_B)
            
            
            midpoint= self.Z / 2

            #z_A = z_A % self.Z
            #z_B = z_B % self.Z

            if seg_A is None or seg_B is None:
                raise ValueError("Particles A or B are out of bounds!")

            distance_AB_x = self.r_MIC_x
            
            
            

            # Logging Particle and Segment Data
            file.write(f"\nParticle A: {A}, z_A: {z_A:.10f}, Segment: {seg_A}, Bounds: ({low_A:.10f}, {high_A:.10f})\n")
            file.write(f"Particle B: {B}, z_B: {z_B:.10f}, Segment: {seg_B}, Bounds: ({low_B:.10f}, {high_B:.10f})\n")

            # Case 0: Particles in the same segment
            if seg_A == seg_B:
                file.write(f"Case 0: Particles are in the same segment.\n")
                file.write(f"Initial Virial: {self.X:.10f} \n")
                file.write(f"New Virial: {self.X:.10f} \n")
                return self.X

            # Case 1: Particle A in a segment, B in the segment above (no PBC)
            if (seg_B == (seg_A + 1)) and abs(z_A-z_B) < 12 :
                a = abs(high_A - z_A)  # Distance from A to the upper boundary
                virial = (a / distance_AB_x) * self.X 
                file.write(f"Case 1: Particle A is in segment {seg_A}, B in {seg_B} (above).\n")
                file.write(f"Distance to boundary: {a:.10f} \n")
                file.write(f"distance_AB_x: {distance_AB_x} \n")
                file.write(f"Initial Virial: {self.X:.10f} \n")
                file.write(f"New Virial: {virial:.10f} \n")
                return 2.0*virial
            
            
            
            # Case 2: Particle A in a segment, B in the segment below (no PBC)
            if (seg_B == (seg_A - 1) ) and abs(z_A-z_B) < 12:
                a = abs(z_A - low_A)  # Distance from A to the lower boundary
                virial = self.X * (a / distance_AB_x)
                file.write(f"Case 2: Particle A is in segment {seg_A}, B in {seg_B} (below).\n")
                file.write(f"Distance to boundary: {a:.10f} \n")
                file.write(f"distance_AB_x: {distance_AB_x} \n")
                file.write(f"Initial Virial: {self.X:.10f} \n")
                file.write(f"New Virial: {virial:.10f} \n")
                return 2.0*virial
            
            
            

            if abs(z_A-z_B) > 12:
                if seg_A == 0 and seg_B == 1:#self.num_segments - 1:
                    a = abs(z_A-low_A)  # Distance from A to the upper boundary of the segment
                    virial = (a / distance_AB_x) * self.X 
                    file.write(f"Case 3: Particle A is in segment {0}, B in {seg_B} (PBC).\n")
                    file.write(f"Distance to boundary: {a:.10f} \n")
                    file.write(f"distance_AB_x: {distance_AB_x} \n")
                    file.write(f"Initial Virial: {self.X:.10f} \n")
                    file.write(f"New Virial: {virial:.10f} \n")
                    return 2.0*virial
                    
                
                
                    

            if abs(z_A-z_B) > 12:
                if seg_B == 0 and seg_A == 1:#self.num_segments - 1:
                    a = abs(high_A - z_A)
                    virial = (a / distance_AB_x) * self.X 
                    file.write(f"Case 4: Particle A is in segment {seg_A}, B in {0} (PBC).\n")
                    file.write(f"Distance to boundary: {a:.10f} \n")
                    file.write(f"distance_AB_x: {distance_AB_x} \n")
                    file.write(f"Initial Virial: {self.X:.10f} \n")
                    file.write(f"New Virial: {virial:.10f} \n")
                    return 2.0*virial
                            


            # Default error handling
            file.write("Error: Undefined case! Something is wrong.\n")
            return None



                    
                    


class Radius_Calculation:
    def __init__(self, 
                box_size,
                segment,
                num_segments,
                u0,  # Water MD universe
                u,   # Mem MD universe 
                atoms_positions_water=None, 
                atoms_positions_mem=None,
                *args, 
                **kwargs):
        super().__init__(*args, **kwargs) 
        self.box_size = box_size
        self.universe_water = u0  
        #print(self.universe_water)
        self.universe_membrane = u
        #print(self.universe_membrane)
        self.atoms_positions_water    = self.universe_water.atoms.positions
        #print("Water positions:\n",self.atoms_positions_water)
        self.atoms_positions_membrane = self.universe_membrane.atoms.positions
        #print("Membrane positions:\n",self.atoms_positions_membrane)
        self.segment = segment
        self.num_segments = num_segments
        

        
    def Radius(self):
        """
        This methos calculates the dynamic cylinder radius and COM 
        """
        cylinder_x_min =(np.min(self.universe_water.positions[:, 0]))
        #print("cylinder_x_min:\n",cylinder_x_min)
        cylinder_x_max =(np.max(self.universe_water.positions[:, 0]))
        #print("cylinder_x_max:\n",cylinder_x_max)
        segment_y_min=-10000
        segment_y_max=+10000
        segment_z_min=-10000
        segment_z_max=+10000
        # Total_length = 337.50332642 ->>> obtained from the box
        Total_length =  337.50332642#(cylinder_x_max-cylinder_x_min)
        #print("Total_length:\n",Total_length)
        segment_width = float(Total_length / self.num_segments)
        #print("segment_width:\n",segment_width)
        segment_x_min = -10000#cylinder_x_min + self.segment * segment_width
        #print("segment_x_min:\n",segment_x_min)
        segment_x_max = +10000#segment_x_min + segment_width
        #print("segment_x_max:\n",segment_x_max)
        
        # Calculate the lipid center of mass for the segment 
        segment_mem_C5 = self.universe_membrane.select_atoms(f"prop x >= {-10000} and prop x < {10000}")
        #print("segment_mem_C5:\n",len(segment_mem_C5))
        center_of_mass_mem = segment_mem_C5.center_of_mass()
        #print("center_of_mass_mem:\n",center_of_mass_mem)
        # Calculate the distances of atoms from the center of mass in the y and z directions (ignoring x)  in a certin segment
        distances = np.linalg.norm(segment_mem_C5.positions[:, 1:3] - center_of_mass_mem[1:3], axis=1) 
        # Calculate mean, inner, and outer radius for this segment
        MIN_RADIUS_SEGMENT  = np.min(distances)
        #print("MEMBRANE MIN RADIUS in Angestrum:\n",MIN_RADIUS_SEGMENT)
        MAX_RADIUS_SEGMENT  = np.max(distances)
        #print("MEMBRANE MAX RADIUS in Angestrum:\n",MAX_RADIUS_SEGMENT)
        # Radius cutoff values (adjusted)
        radius_cutoff_inner = 15#MIN_RADIUS_SEGMENT   # Angstroms
        radius_cutoff_outer = 15#MAX_RADIUS_SEGMENT   # Angstroms
        # Box volume inside and outside the cylinder (Angestrum)
        x_min=0
        x_max=self.box_size[0]
        #print("x_max:",x_max)
        y_min=0
        y_max=self.box_size[1]
        #print("y_max:",y_max)
        z_min=0
        z_max=self.box_size[2]
        #print("z_max:",z_max)
        box_volume_inside = np.pi * (MIN_RADIUS_SEGMENT ** 2) * (x_max - x_min)
        #print("box_volume_inside:",box_volume_inside)
        # box_volume_outside=volume of the box−volume of the large cylinder
        box_volume_outside = ((y_max - y_min) * (z_max - z_min) * (x_max - x_min)) - np.pi * (MAX_RADIUS_SEGMENT ** 2) * (x_max - x_min)
        #print("box_volume_outside:",box_volume_outside)
        # Need to select water particles within the cylinder radius (MIN_RADIUS_SEGMENT)
        segment_water = self.universe_water.select_atoms(f"prop x >= {-10000} and prop x < {10000}")  # Water partilces in corresponding segment 
        #print("segment_water selected in Radius:\n",segment_water) 
        #print("atom positions in Radius:\n",segment_water.positions) 
        return segment_water,center_of_mass_mem,radius_cutoff_inner, radius_cutoff_outer, box_volume_inside, box_volume_outside
        
        



      
class InitializeSimulation:
    def __init__(self, 
                 u,  # MD universe
                 atoms_positions=None,  # Array - Angstroms
                 box_size=None,  # Default to None, or replace with a specific default value
                 num_atoms=None, 
                 cutoff=None,  # Default to None, or specify a default value
                 matrix=None,  # Default to None, or specify a default value
                 p_ideal_atm=None,
                 indices =None,
                 inner_radius=None,
                 outer_radius=None,
                 segment=None,
                 center_x =None,
                 center_y =None,
                 center_z=None,
                 box_volume_inside =None,
                 box_volume_outside=None,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)  
        

        self.universe = u  # MDAnalysis universe
        #print("segment_water selected in InitializeSimulation:\n",self.universe )
        #print("atoms_positions in InitializeSimulation:\n",self.universe.positions )
        self.segment =segment
        self.center_x =center_x,
        self.center_y =center_y,
        self.center_z=center_z,
        self.atoms_positions = atoms_positions
        self.box_size = box_size
        self.num_atoms = num_atoms
        self.cutoff = cutoff
        self.matrix = matrix
        self.p_ideal_atm=p_ideal_atm
        self.indices= indices
        self.outer_radius=outer_radius
        self.inner_radius=inner_radius
        self.box_volume_inside=box_volume_inside
        self.box_volume_outside=box_volume_outside
        
        


        
        
    def Box_Average(self):
        volume = np.prod(self.box_size[:3])   # box volume
        return volume 
        

    def p_ideal(self):
        velocity = self.universe.select_atoms('name W').velocities  
        volume = np.prod(self.box_size[:3])   # box volume
        kinetic_energy =0.5* sum((atom_mass * np.dot(atom.velocity, atom.velocity)) for atom in ag)
        #print("kinetic_energy (amu)(A/ps)^2:",kinetic_energy)
        self.p_ideal = (2 * kinetic_energy) / (3 * volume)
        #print("p_ideal amu / A ps^2 \n",self. p_ideal)
        self.p_ideal_atm = self.p_ideal * 163.8
        return self.p_ideal_atm

        
        
    def calculate_pressure(self):
        """Evaluate calculate_pressure based on the Virial"""
        # Compute the ideal contribution
        #p_ideal_atm =  self. p_ideal_atm
        #print("p_ideal_atm:\n",p_ideal_atm)
        #print("Subset contact matrix shape in pressure:", self.matrix.shape)
        num_atoms = len(self.indices)
        #print("num_atoms in calc:",num_atoms)
        #print("Phase 5")
        #print("SUB Matrix in INIT",matrix.shape)
        MyVirial = []
        MyVirial_KJ_MOL = []
        pressures = [] 
        neighbor_lists = []
        output_file = f"atom_positions_and_neighbors_Inside_{self.segment}_data.txt"
        # Open the file in append mode
        with open(output_file, "a") as f:
            Indices = (self.indices)
            # Keep track of processed pairs to avoid double counting
            processed_pairs = set()

            for index, row in df_neighbors.iloc[:-1].iterrows():
                selected_atom = row['selected_index']
                neighbors = row['neighbor_indices']
                
                # Remove self-neighboring atoms
                neighbors = [neighbor for neighbor in neighbors if neighbor != selected_atom]
                
                # Process neighbors
                filtered_neighbors = []
                for neighbor in neighbors:
                    # If the reverse pair (neighbor, selected_atom) has already been processed, skip it
                    if (selected_atom, neighbor) not in processed_pairs and (neighbor, selected_atom) not in processed_pairs:
                        filtered_neighbors.append(neighbor)
                        # Mark the pair as processed
                        processed_pairs.add((selected_atom, neighbor))
                
                #print(f"Atom {selected_atom} has neighbors: {filtered_neighbors}")
                
                
                neighbor_lists.append(filtered_neighbors)

                # Get positions of the current atom and its neighbors 
                current_atom_position = self.universe[self.indices[index]].position
                #print("current_atom_position:\n",current_atom_position)
                f.write(f"current_atom_position: {self.universe[self.indices[index]].position}\n")
                #print("current_atom_position in class:",current_atom_position)
                # Get positions of neighbors
                #neighbor_positions = [self.universe.atoms[Indices[neighbor]].position.tolist() for neighbor in filtered_neighbors]
                neighbor_positions = [self.universe[self.indices[ele]].position for ele in filtered_neighbors]
                #print("neighbor_positions:\n",neighbor_positions)
                f.write(f"neighbor_positions: {neighbor_positions}\n\n")
                        
                        
                
                distances = [np.linalg.norm(current_atom_position - neighbor_position) for neighbor_position in neighbor_positions]
                f.write(f"distances: {distances}\n\n")
                displacement = [(current_atom_position - neighbor_position) for neighbor_position in neighbor_positions]
                f.write(f"displacement: {displacement}\n\n")
                #print(f"Atom {self.indices[cpt]} position: {current_atom_position}")
                #print(f"Neighbors of atom {self.indices[cpt]}: {list}")
                #f.write(f"Neighbors of atom {self.indices[cpt]}: {list}\n")
                #print(f"Neighbor positions: {neighbor_positions}")
                #print(f"displacement: {displacement}")
                #print(f"Norm of displacement vector (Before MIC): {distances}")
                #print("\n\n\n")
                
            #print("neighbor_lists:",neighbor_lists)
            
            #print("\n\n\n")
            
            
        
          
        f_vdw_switch_count = 0  
        output_file = f"All_interactions_{self.segment}_data.txt"
        # Open the file in write mode
        with open(output_file, 'w') as f:
            total_virials_all_particles = []
            for Ni in np.arange(np.sum(num_atoms)-1):
                #print("Ni:\n",Ni)
                position_of_i = self.universe[self.indices[Ni]].position
                position_of_A = position_of_i
                #print(f"position_of_A\n: {position_of_A}")
                
                
                #print(f"Atom {Ni} position: {position_of_i}")
                # Get the neighbor list for the current atom
                neighbors_of_i =  neighbor_lists[Ni]
                #print("neighbors_of_i:",neighbors_of_i)
                neighbor_positions = [self.universe[neighbor].position for neighbor in neighbors_of_i]
                #print(f"Positions of neighbors for atom {Ni}: {neighbor_positions}")
                
                # Calculate displacement vectors for each neighbor
                total_virial = 0
                for neighbor in neighbors_of_i:
                    neighbor_position = self.universe[neighbor].position
                    position_of_B = neighbor_position
                    x_position_of_B = position_of_B[0]
                    # Displacement vector calculation
                    displacement_vector = neighbor_position - position_of_i
                    norm_displacement_vector = np.linalg.norm(displacement_vector)
                    # Apply Minimum Image Convention (MIC)
                    displacement_vector_mic = displacement_vector - np.round(displacement_vector / box_boundaries) * box_boundaries
                    # Calculate norm of the MIC-adjusted displacement vector
                    norm_displacement_vector_mic = np.linalg.norm(displacement_vector_mic)                    
                    r_MIC = norm_displacement_vector_mic  # Angstrom
                    if r_MIC > 12:
                        print("Something went wrong!")

                    # Calculate force_ij using the van der Waals potential switch
                    f_vdw = F_vdw_switch(np.array([r_MIC]))[0]
                    f_vdw_switch_count += 1  # Increment the counter
                    # Calculate Virial
                    Virial = norm_displacement_vector_mic * f_vdw  # kCal/mol
                    
                    # Radial Irving-Kirwood method
                    ### for the pressure outside, we take the maximum membrane radius 
                    cylinder_radius = self.inner_radius
                    cutoff=12
                    RMax=self.inner_radius+cutoff
                    #print("RMax:",RMax)
                    # Check positions of particles A and B
                    A = position_of_A
                    B = position_of_B
                                   

                    # Initialize circle plot object
                    circle = CirclePlot(
                        origin_yy=self.center_y,
                        origin_zz=self.center_z,
                        cylinder_radius=cylinder_radius,
                        cutoff=cutoff,
                        RMax=RMax
                    )
                    
                    # check_position ---> R
                    # check_position_inside_cutoff ---> R+d
                    # check_position_inside_sharp_cutoff ---> Rmax=sqrt(R**2+d**2)
                    
                    Distance_A_To_B = np.sqrt((A[1] - B[1])**2 + (A[2] - B[2])**2)
                    #intersections = circle.compute_intersection(A, B)
                    #count_non_empty_tuples = len([x for x in intersections if x])

                    #print(Distance_A_To_B)
                    
                
                    if (circle.check_position_inside_cutoff(A[1], A[2])) == "above" and  (circle.check_position_inside_cutoff(B[1], B[2])) == "above":
                        intersections = circle.compute_intersection(A, B)
                        S_1, S_2 = 0, 0
                        New_Virial = 0
                        total_virial += New_Virial
                        DELTA_S=abs(S_1-S_2)
                        #circle.plot_circle(A, B, S_1, S_2, DELTA_S, Virial, New_Virial)
                        continue  
                    
                
                    
  

                    # If none of the above scanrios happens, then go ahead with scanrio 5.    
                    # Scenario 5: Both particles are inside the cutoff radius (R+d), so further checks are needed
                    if (circle.check_position_inside_cutoff(A[1], A[2])) == "below" and  (circle.check_position_inside_cutoff(B[1], B[2])) == "below":
                        if Distance_A_To_B > 12:
                            print("Ops. PBC. Sth wrong !")
                        #print("YES. Both particles are inside the Maximum allowd radius, i.e, Cylinder_Radius+Cutoff_Radius.")
                        intersections = circle.compute_intersection(A, B)
                        #print("intersections",intersections)
                        count_non_empty_tuples = len([x for x in intersections if x])
                        #print("# OF Intersections:",count_non_empty_tuples)

                        if not any(intersection != () for intersection in intersections): # ---> CHECKED !
                            S_1, S_2 = 0, 0
                            DELTA_S=abs(S_1-S_2)
                            #print(f"No solution found and zero contribution to the virial of particles inside the cyllinder.")
                            New_Virial = 0 
                            #circle.plot_circle(A, B, S_1, S_2, DELTA_S, Virial, New_Virial)
                        else:
                            #print("A valid intersection exists, proceed with calculations.")
                            # CASE 1: ONE INSIDE AND ONE OUTSIDE 
                            if (circle.check_position(A[1], A[2])) == "below" and (circle.check_position(B[1], B[2])) == "above":
                                #print("Particle A below the cylinder radius and Particle B above the radius")
                                #Distance_A_To_B = np.sqrt((A[1] - B[1])**2 + (A[2] - B[2])**2)
                                #print("Distance_A_To_B:\n", Distance_A_To_B)
                                S_1, S_2 = intersections[0][0][0], intersections[1][0][0]
                                #print(intersections)
                                if 0 <= S_1 <= 1: # ---> CHECKED !
                                    #print("S_1:",S_1)
                                    #print("First solution has a valid S:0<S<1.")
                                    #Distance_To_boundary = np.sqrt((A[1] - Intersections[0][1])**2 + (A[2] - Intersections[0][2])**2)
                                    #New_Virial = (Distance_To_boundary / Distance_A_To_B) * Virial
                                    New_Virial = S_1 * Virial
                                    #DELTA_S=abs(S_1-S_2)
                                    #circle.plot_circle(A, B, S_1, S_2, DELTA_S, Virial, New_Virial)
                                else: # ---> CHECKED !
                                    #print("S_2:",S_2)
                                    #print("Second solution has a valid S:0<S<1.")
                                    #Distance_To_boundary = np.sqrt((A[1] - Intersections[1][1])**2 + (A[2] - Intersections[1][2])**2)
                                    #New_Virial = (Distance_To_boundary / Distance_A_To_B) * Virial
                                    New_Virial = S_2 * Virial
                                    #print("New_Virial:\n", New_Virial)
                                    #DELTA_S=abs(S_1-S_2)
                                    #circle.plot_circle(A, B, S_1, S_2, DELTA_S, Virial, New_Virial)
                                    
                                    
                            if (circle.check_position(A[1], A[2])) == "above" and (circle.check_position(B[1], B[2])) == "below":
                                #print("Particle A below the cylinder radius and Particle B above the radius")
                                #Distance_A_To_B = np.sqrt((A[1] - B[1])**2 + (A[2] - B[2])**2)
                                #print("Distance_A_To_B:\n", Distance_A_To_B)
                                S_1, S_2 = intersections[0][0][0], intersections[1][0][0]
                                #print(intersections)
                                if 0 <= S_1 <= 1: # ---> CHECKED !
                                    #print("S_1:",S_1)
                                    #print("First solution has a valid S:0<S<1.")
                                    #Distance_To_boundary = np.sqrt((A[1] - Intersections[0][1])**2 + (A[2] - Intersections[0][2])**2)
                                    #New_Virial = (Distance_To_boundary / Distance_A_To_B) * Virial
                                    New_Virial = S_1 * Virial
                                    #DELTA_S=abs(S_1-S_2)
                                    #circle.plot_circle(A, B, S_1, S_2, DELTA_S, Virial, New_Virial)
                                else: # ---> CHECKED !
                                    #print("S_2:",S_2)
                                    #print("Second solution has a valid S:0<S<1.")
                                    #Distance_To_boundary = np.sqrt((A[1] - Intersections[1][1])**2 + (A[2] - Intersections[1][2])**2)
                                    #New_Virial = (Distance_To_boundary / Distance_A_To_B) * Virial
                                    New_Virial = (1-S_2) * Virial
                                    #print("New_Virial:\n", New_Virial)
                                    #DELTA_S=abs(S_1-S_2)
                                    #circle.plot_circle(A, B, S_1, S_2, DELTA_S, Virial, New_Virial)
                                    
                                    
                                    

                                    
                                    
                            # CASE 0: BOTH INSIDE ---> CHECKED !
                            elif (circle.check_position(A[1], A[2])) == "below" and (circle.check_position(B[1], B[2])) == "below":
                                #print("Particle A below the cylinder radius and Particle B below the cylinder radius with no solutions.")
                                #print("Virial does not change!.")
                                New_Virial = Virial  
                                #S_1=S_2=DELTA_S=0
                                #circle.plot_circle(A, B, S_1, S_2,DELTA_S,Virial, New_Virial)
                                #print("New_Virial:\n", New_Virial)
                                
                                
                            # CASE 2: BOTH OUTSIDE BUT STILL CONTRIBUTIONG
                            elif (circle.check_position(A[1], A[2])) == "above" and (circle.check_position(B[1], B[2])) == "above":
                                S_1, S_2 = intersections[0][0][0], intersections[1][0][0]
                                #circle.plot_circle(A, B, S_1, S_2)
                                if 0 <= S_1 <= 1 and 0 <= S_2 <= 1: # ---> CHECKED !
                                    #print("Particle A and B both above the cylinder radius but both inside the cutoff radius and therefore contributing!")
                                    DELTA_S = abs(S_1 - S_2)
                                    #print("DELTA_S:",DELTA_S)
                                    New_Virial = DELTA_S * Virial
                                    #circle.plot_circle(A, B, S_1, S_2, DELTA_S, Virial, New_Virial)
                                
                                else: # ---> CHECKED !
                                    #print("Particle A and B both above the cylinder radius but no intersection !")
                                    #print("Virial:",Virial)
                                    New_Virial = 0
                                    #print("New_Virial:",New_Virial)
                                    #DELTA_S = abs(S_1 - S_2)
                                    #circle.plot_circle(A, B, S_1, S_2,DELTA_S,Virial, New_Virial)

                                    
                                    
                                
                    # ---> CHECKED !        
                    else:
                        #print("Particle is on the cutoff radius or pair particles are not inside the Maximum allowd radius, i.e, Cylinder_Radius+Cutoff_Radius.")
                        New_Virial = 0
                        #DELTA_S=0
                        #S_1=S_2=0
                        #circle.plot_circle(A, B, S_1, S_2,DELTA_S,Virial, New_Virial)
                            


                    # Accumulate virial contribution **after all conditions**
                    #print("New_Virial:",New_Virial)
                    total_virial += New_Virial  
                    #print("total_virial:",total_virial)
                    #print("\n")
                    
                # Append total virial for this atom to the list
                total_virials_all_particles.append(total_virial)
                #print("total_virials_all_particles:",(total_virials_all_particles))


        ## Calculate pressure for the current frame
        #print(f"F_vdw_switch was executed {f_vdw_switch_count} times.")
        #print(total_virials_all_particles)
        CONVERSION_TO_ATM = 69475.55667#4184*16.38855174 
        Sigma_Virial = (np.sum(total_virials_all_particles)) # kCal/mol
        print("Sigma_Virial:\n",Sigma_Virial)
        #print(f"Virial in kJ/mol: {0.5*(Sigma_Virial*4.184):.10f}") # This is consistant with gromacs output for two particles when (Vxx+Vyy+Vzz)/2
        Virial_KJ_Per_MOL=0.5*(Sigma_Virial*4.184)
        #print(f"Virial in kCal/mol: {0.5*(Sigma_Virial):.10f}")
        Virial_Kcal_Per_MOL=0.5*(Sigma_Virial)
        Volume = box_boundaries[0] * box_boundaries[1] * box_boundaries[2] # Angestrum^3
        #Virial = ( (np.sum(total_virials_all_particles)) / (3*Volume))   # kCal/mol.Angestrum^3 
        #print("Pressure in kcal/mol·Å³ :",Virial)
        #print(self.box_volume_inside)
        Pressure = ( (np.sum(total_virials_all_particles)) / (3*self.box_volume_inside) ) * CONVERSION_TO_ATM  #  atm 
        #print(f"Pressure in atm: {Pressure:.10f}")
        MyVirial.append(Sigma_Virial)
        MyVirial_KJ_MOL.append(Virial_KJ_Per_MOL)
        pressures.append(Pressure)
        
        return MyVirial , pressures , MyVirial_KJ_MOL
         
                





Kinetic_values=[]            
Volume_values=[]
Virial_values_outside=[]
pressure_values_outside=[]
Virial_values_inside=[]
pressure_values_inside=[]
universe  = mda.Universe("water_only.tpr", "water_only.trr")
universe2 = mda.Universe("mem_only.tpr"  , "mem_only.trr")
mean_segment_virial   = {f'Segment_{i+1}': [] for i in range(num_segments)}
mean_segment_pressure = {f'Segment_{i+1}': [] for i in range(num_segments)}
Myvirial=[]
Mypressure=[]
Myvirial_KJ_PER_MOL=[]

#for ts, ts2 in zip(universe.trajectory[index-1:index], universe2.trajectory[index-1:index]):
for ts, ts2 in zip(universe.trajectory[:1], universe2.trajectory[:1]): 
    # Processing frames
    print(f"Processing Frame in water simulation: {ts.frame}")
    #print(f"Processing Frame in membrane simulation: {ts2.frame}")
    cutoff = 12  #  cutoff in Angstroms
    box_boundaries = universe2.dimensions[:3]
    box_geometry = np.array([90, 90, 90])  # Cubic box
    box_size = np.array(box_boundaries.tolist()+box_geometry.tolist())
    #print("box_size:",box_size)
    ag_water = universe.select_atoms('all')
    #print("Water Total number:",len(ag_water))
    #print(ag_water.positions[1])
    #print(ag_water.positions[35])
    ag_mem   = universe2.select_atoms('all') 
    #print("Membrane Total number:",len(ag_mem))
    #print(ag_mem.positions)
    # cylinder_x_min = np.round(np.min(ag_water.positions[:, 0]), 10)
    # #print("cylinder_x_min:\n",cylinder_x_min)
    # cylinder_x_max = np.round(np.max(ag_water.positions[:, 0]), 10)
    # #print("cylinder_x_max:\n",cylinder_x_max)
    # Total_Length = cylinder_x_max - cylinder_x_min 
    # #print("Total_Length in Main:\n",Total_Length)
    # Loop over each segment
    for segment in range(num_segments):
        #print("segment number:\n",segment)
        #print("Phase 1")
        radius_calculator = Radius_Calculation(box_size, segment, num_segments , u0=ag_water, u=ag_mem, atoms_positions_water=None , atoms_positions_mem=None)
        segment_ag , center_of_mass ,  inner_radius, outer_radius , box_volume_inside, box_volume_outside = radius_calculator.Radius()
        #print("segment_water selected in Main:\n",segment_ag)
        cutoff=12
        # inner_radius+cutoff
        # Calculates the Euclidean distance in the YZ-plane (i.e., the radial distance from the cylinder axis).
        selected_atoms_inside = segment_ag[[np.linalg.norm(atom.position[1:3] - center_of_mass[1:3]) < 10000000000000  for atom in segment_ag]]
        #print("Water Particles Inside Cylinder: \n",len(selected_atoms_inside))
        selected_atoms_inside_positions = np.array([atom.position for atom in selected_atoms_inside])
        #print("BBBB:",len(selected_atoms_inside_positions))
        # Plot selected particles within the cylinder
        center_x = center_of_mass[0]  # X-coordinate of the membrane center of mass
        center_y = center_of_mass[1]  # Y-coordinate of the membrane center of mass 
        center_z = center_of_mass[2]  # Z-coordinate of the membrane center of mass 
        Inner_radius = inner_radius
        # Extract X, Y, Z coordinates
        # x = selected_atoms_inside_positions[:, 0]
        # y = selected_atoms_inside_positions[:, 1]
        # z = selected_atoms_inside_positions[:, 2]
        # # Create a 3D plot
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # theta = np.linspace(0, 2 * np.pi, 100) 
        # y_circle = center_y + Inner_radius * np.cos(theta)  # Y-coordinates of the circle
        # z_circle = center_z + Inner_radius * np.sin(theta)  # Z-coordinates of the circle
        # x_circle = np.full_like(theta, center_x)      # Keep X fixed (YZ plane)
        # ax.plot(x_circle, y_circle, z_circle, color='r', linewidth=10, label="outer_radius Radius")
        # #ax.scatter(center_x, center_y, center_z, color='red', s=100, label="Center of Mass")
        # ax.scatter(x, y, z, c='b', marker='o', alpha=0.5, label="Selected Water Molecules")
        # ax.set_xlabel("X-axis")
        # ax.set_ylabel("Y-axis")
        # ax.set_zlabel("Z-axis")
        # ax.set_xlim([0, 90])  
        # ax.set_ylim([0, 90])  
        # ax.set_zlim([0, 90])   
        # filename = "water_molecules_inside_cylinder_"+str(ts.frame)+".png"
        # plt.savefig(filename, dpi=300, bbox_inches='tight')
        #plt.show()
        selected_atom_inside_indices = [atom.index for atom in selected_atoms_inside] # ---> Using these indices, one can select the inside positions from the the universe (ag0) 
        #print("CCCC:",(selected_atom_inside_indices))
        selected_atoms_inside_ag = universe.atoms[selected_atom_inside_indices]
        #print("DDDD:",len(selected_atoms_inside_ag))
        selected_atoms_inside_ag_positions = np.array([atom.position for atom in selected_atoms_inside_ag])
        #print("selected_atoms_inside_ag_positions:",selected_atoms_inside_ag_positions)
        #print(f"Segment {segment}:")
        #print(f"# of particles taken into consideration: {len(selected_atoms_inside)} atoms")
        #from MDAnalysis.analysis import distances
        # Compute the contact matrix  for "all" particles in the system 
        
        # Compute contact pairs
        
        selected_pos = ag_water.positions[selected_atom_inside_indices]  #  atoms of interest
        #print("selected_pos:\n",(selected_pos))
        all_pos = ag_water.positions                                     #  potential neighbors 
        #print("all_pos:\n",(all_pos))
                
                
        
        # Compute capped distances (only pairs within cutoff, no distances)
        result = capped_distance(
            selected_pos,
            all_pos,
            max_cutoff=cut_off,
            box=box_size,
            return_distances=False
        )
        
        


        # Convert to (selected_index -> list of neighbor_indices)
        neighbor_dict = defaultdict(list)
        for i, j in zip(*result) if isinstance(result, tuple) else result:
            neighbor_dict[i].append(j)

        # Convert to DataFrame
        df_neighbors = pd.DataFrame([
            {'selected_index': k, 'neighbor_indices': v}
            for k, v in neighbor_dict.items()
        ])

        
        df_neighbors.to_csv('neighbors_list_Inside.txt', index=False, sep='\t')
        
        
       
        
        
        # matrix = distances.contact_matrix(
        #     ag_water.positions,  
        #     cutoff=cut_off,
        #     returntype="numpy",
        #     box=box_size  
        # )
        #print("\n\n\n\n\n")
        #print("Matrix\n",len(matrix))
        # Particles in the corresponding segment & inside the cylinder 
        #matrix = matrix[selected_atom_inside_indices,:]
        #print("Sub Matrix\n",len(matrix))
        #print("Matrix",matrix.shape)
        # Transpose the positions array: converts from (N, 3) to (3, N)
        
        
        
        
        
        
        
        
        ag0 = universe.select_atoms('all') 
        num_atoms = len(ag0)
        P = ag0.positions   
        ag0_transposed = P.T 
        #Create an instance of InitializeSimulation
        simulation = InitializeSimulation(
            u = ag_water,                       # all water particles in  (N, 3) 
            atoms_positions=ag0_transposed,     # all water particle positions in  (3, N)
            box_size=box_boundaries,            # Box size
            num_atoms=num_atoms,                # all particles
            cutoff=cutoff,
            matrix = df_neighbors, 
            indices = selected_atom_inside_indices,
            inner_radius= inner_radius,
            outer_radius= outer_radius,
            segment=segment,
            center_x =center_x,
            center_y =center_y,
            center_z=center_z,
            box_volume_inside = box_volume_inside,
            box_volume_outside = box_volume_outside,
        )
        
        #kinetic = simulation.p_ideal()
        #Kinetic_values.append(kinetic)
        Volume = simulation.Box_Average()
        Volume_values.append(Volume)
        pressure = simulation.calculate_pressure()
        Myvirial.append(pressure[0])
        Mypressure.append(pressure[1])
        Myvirial_KJ_PER_MOL.append(pressure[2])
        Virial_values_inside.append(pressure[0])
        pressure_values_inside.append(pressure[1])
        mean_segment_virial[f'Segment_{segment+1}'].append(Virial_values_inside)
        mean_segment_pressure[f'Segment_{segment+1}'].append(pressure_values_inside)
        
        
        
        
        output_filename = f"Inside_{index-1}.txt"
        with open(output_filename, "w") as f:
            f.write(f"{pressure[0][0]}\n")
        
        # output_filename = f"Outside_Cylinder_virial_KJ_PER_MOL_{index-1}.txt"
        # with open(output_filename, "w") as f:
        #     f.write(f"{pressure[0]}\n")
            
#         output_filename = f"Outside_Cylinder_Pressure_atm_{index-1}.txt"
#         with open(output_filename, "w") as f:
#             f.write(f"{pressure[0]}\n")
#             
#             
#         output_filename = f"Outside_Cylinder_Virial_Kcal_per_mol_{index-1}.txt"
#         with open(output_filename, "w") as f:
#             f.write(f"{pressure[0]}\n")
#         
# 
#         





# End the timer
end_time = time.time()
elapsed_time = end_time - start_time
#print(f"Total execution time for outside virial for {index-1}: {elapsed_time:.6f} seconds")
print(f"End of {index-1}")






