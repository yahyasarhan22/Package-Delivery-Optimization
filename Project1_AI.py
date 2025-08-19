# Yahya Sarhan      1221858                     Dr.Yazan Abu Farha                                                 sec:1
import math
import random
import copy
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#=============================== Create Random Solution ===========================
def create_random_solution(vehicles_cap, packages):
    solution = {v_id: [] for v_id in vehicles_cap}
    vehicle_loads = {v_id: 0 for v_id in vehicles_cap}#inital load zero
    package_indices = list(range(len(packages)))
    random.shuffle(package_indices) #to get random packages 

    for idx in package_indices:
        pkg = packages[idx]# get a random package
        vehicle_ids = list(vehicles_cap.keys())
        random.shuffle(vehicle_ids) # get a random vehicle
        for v_id in vehicle_ids:
            if vehicle_loads[v_id] + pkg["weight"] <= vehicles_cap[v_id]:
                solution[v_id].append(idx)
                vehicle_loads[v_id] += pkg["weight"]
                break
    return solution

#==================================== Calculate Total Distance ==========================

def calculate_total_distance(solution, packages):
    total_distance = 0.0
    total_priority_penalty = 0.0


    for vehicle_id, package_indices in solution.items():
        current_x, current_y = 0, 0
        for pkg_idx in package_indices:
            pkg = packages[pkg_idx]
            # Distance from current location to next location
            dist = math.sqrt((pkg["x"] - current_x)**2 + (pkg["y"] - current_y)**2)

            # Priority penalty: lower priority (higher number) = more penalty
            priority_weight = 0.2 / pkg["priority"]
            penalty = dist * priority_weight

            total_distance += dist
            total_priority_penalty += penalty

            current_x, current_y = pkg["x"], pkg["y"] #update the location

    objective = total_distance + total_priority_penalty
    return total_distance,objective

#============================= Neighbor =========================================

def get_neighbor(solution, vehicles_cap, packages):
    new_solution = copy.deepcopy(solution)
    vehicle_ids = list(vehicles_cap.keys())
    vehicle_loads = {v_id: sum(packages[i]["weight"] for i in new_solution[v_id]) for v_id in vehicle_ids}

    operation = random.randint(0, 2)

    if operation == 0:
        # Swap two packages within the same vehicle
        non_empty_vehicles = [v for v in vehicle_ids if len(new_solution[v]) >= 2]
        if not non_empty_vehicles:
            return new_solution
        v = random.choice(non_empty_vehicles)
        pkg1, pkg2 = random.sample(new_solution[v], 2)
        idx1 = new_solution[v].index(pkg1)
        idx2 = new_solution[v].index(pkg2)
        new_solution[v][idx1], new_solution[v][idx2] = new_solution[v][idx2], new_solution[v][idx1]

    elif operation == 1:
        # Move one package from one vehicle to another
        non_empty_vehicles = [v for v in vehicle_ids if len(new_solution[v]) > 0]
        if not non_empty_vehicles:
            return new_solution
        from_v = random.choice(non_empty_vehicles)
        pkg = random.choice(new_solution[from_v])
        other_vehicles = [v for v in vehicle_ids if v != from_v]
        if not other_vehicles:
            return new_solution  # No other vehicle to move to
        to_v = random.choice(other_vehicles)
        if vehicle_loads[to_v] + packages[pkg]["weight"] <= vehicles_cap[to_v]:
            new_solution[from_v].remove(pkg)
            new_solution[to_v].append(pkg)

    else:
        # Two-way swap between two vehicles
        vehicles_with_packages = [v for v in vehicle_ids if len(new_solution[v]) > 0]
        if len(vehicles_with_packages) < 2:
            return new_solution
        v1, v2 = random.sample(vehicles_with_packages, 2)
        pkg1 = random.choice(new_solution[v1])
        pkg2 = random.choice(new_solution[v2])

        # Check if both swaps are valid (capacity)
        v1_new_load = vehicle_loads[v1] - packages[pkg1]["weight"] + packages[pkg2]["weight"]
        v2_new_load = vehicle_loads[v2] - packages[pkg2]["weight"] + packages[pkg1]["weight"]

        if v1_new_load <= vehicles_cap[v1] and v2_new_load <= vehicles_cap[v2]:
            new_solution[v1].remove(pkg1)
            new_solution[v1].append(pkg2)
            new_solution[v2].remove(pkg2)
            new_solution[v2].append(pkg1)

    return new_solution

#================================ Simulated Annealing ===============================

def simulated_annealing(vehicles_cap, packages, initial_temp=1000, cooling_rate=0.95, stop_temp=1, iterations_per_temp=100):
    #create a random solution for current state
    current_solution = create_random_solution(vehicles_cap, packages)
    _,current_cost = calculate_total_distance(current_solution, packages)
    best_solution = current_solution
    _,best_cost = calculate_total_distance(best_solution,packages)
    T = initial_temp #1000

    while T > stop_temp:
        for _ in range(iterations_per_temp):#100
            neighbor = get_neighbor(current_solution, vehicles_cap, packages)
            _,neighbor_cost = calculate_total_distance(neighbor, packages)
            delta = neighbor_cost - current_cost
            # Decide to accept or not
            if delta < 0 or random.random() < math.exp(-delta / T):#random.uniform(0, 1) => get randon float number between 0 and 1(not inclusive) 
                current_solution = neighbor
                current_cost = neighbor_cost
                #check if this moving package is the best or not
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost
        T *= cooling_rate
    best_distance,_ =calculate_total_distance(best_solution,packages)

    return best_solution, best_distance

# ============================ Genetic Algorithm Setup ============================

# ============================ Generate Initial Population==============================
 #create N random solutions
def generate_initial_population(pop_size, vehicles_cap, packages):
    return [create_random_solution(vehicles_cap, packages) for _ in range(pop_size)]
# ============================= Calculate Fitness ============================
#Computes how "good" a solution is using your cost function
def fitness(solution, packages):
    _,obj = calculate_total_distance(solution, packages)
    return obj

# =============================== Tournament Selection ==========================
#Randomly picks k solutions, and returns the one with the lowest cost.
def tournament_selection(population, packages, k=3):
    selected = random.sample(population, k)
    selected.sort(key=lambda sol: fitness(sol, packages))
    return selected[0]

# ================================== Crossover ====================================
def crossover(parent1, parent2, vehicles_cap, packages):
    child = {v_id: [] for v_id in vehicles_cap}#Start with an empty solution
    assigned = set()

    # Combine package sources from both parents
    all_assignments = []
    for v_id in vehicles_cap: #Ex:[(0, 1, 'p1'), (1, 1, 'p1'), (2, 2, 'p1'), (1, 2, 'p2'), (3, 1, 'p2')]
        for pkg in parent1[v_id]:
            all_assignments.append((pkg, v_id, 'p1'))
        for pkg in parent2[v_id]:
            all_assignments.append((pkg, v_id, 'p2'))

    random.shuffle(all_assignments)  # Randomize source order

    for pkg_idx, v_id, _ in all_assignments:
        if pkg_idx in assigned:#package is already assigned 
            continue
        #Check every vehicle
        for target_v in vehicles_cap:
           # If the vehicle can carry it => assign it and mark as used
           #If no vehicle can carry it, the package is skipped
            current_load = sum(packages[i]["weight"] for i in child[target_v])
            if current_load + packages[pkg_idx]["weight"] <= vehicles_cap[target_v]:
                child[target_v].append(pkg_idx)
                assigned.add(pkg_idx)
                break

    return child

#============================= Mutate =========================================

def mutate(solution, vehicles_cap, packages, mutation_rate=0.05):
    new_solution = copy.deepcopy(solution)
    #get list of all packages
    all_packages = [pkg_idx for pkgs in new_solution.values() for pkg_idx in pkgs]
    #Get list of vehicle ID
    vehicle_ids = list(vehicles_cap.keys())
    # Loop through each package
    for pkg_idx in all_packages:
        if random.random() < mutation_rate:#With mutation_rate = 0.05, this gives a 5% chance that a package will be considered for mutation
            #Find the current vehicle  holding this package            
            from_v = next((v for v in new_solution if pkg_idx in new_solution[v]), None)
            if from_v is None:
                continue
            #Choose a random different vehicle
            to_v = random.choice(vehicle_ids)
            if from_v == to_v:# skip if same vehicle 
                continue 
            #check if there is avaliable space
            to_v_load = sum(packages[i]["weight"] for i in new_solution[to_v])
            if to_v_load + packages[pkg_idx]["weight"] <= vehicles_cap[to_v]:
                # moves the package to the new vehicle
                new_solution[from_v].remove(pkg_idx)
                new_solution[to_v].append(pkg_idx)
    return new_solution

#========================================= Main Genetic Algorithm ======================================

def genetic_algorithm(vehicles_cap, packages, pop_size=50, mutation_rate=0.05, generations=500):
    #Initial Population
    population = generate_initial_population(pop_size, vehicles_cap, packages)
    #get the best solution from N random solutions
    best_solution = min(population, key=lambda sol: fitness(sol, packages))
    best_cost = fitness(best_solution, packages)
    #Repeat for N Generations
    for _ in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, packages) 
            parent2 = tournament_selection(population, packages)
            child = crossover(parent1, parent2, vehicles_cap, packages)
            child = mutate(child, vehicles_cap, packages, mutation_rate)
            new_population.append(child)

        population = new_population
        #Track the Best Solution
        current_best = min(population, key=lambda sol: fitness(sol, packages))
        current_best_cost = fitness(current_best, packages)
        if current_best_cost < best_cost:
            best_solution = current_best
            best_cost = current_best_cost
    best_distance,_=calculate_total_distance(best_solution,packages)
    return best_solution, best_distance

#==================================== UI ===========================================
class DeliveryOptimizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Package Delivery Optimizer")
        self.root.geometry("800x600")
        self.file_path = tk.StringVar()
        self.algorithm = tk.StringVar(value="Simulated Annealing")
        self.setup_widgets()

    def setup_widgets(self):
        tk.Label(self.root, text="Input File:").pack(pady=5)
        file_frame = tk.Frame(self.root)
        file_frame.pack()
        tk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, padx=5)
        tk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)

        tk.Label(self.root, text="Choose Algorithm:").pack(pady=10)
        algo_menu = ttk.Combobox(self.root, textvariable=self.algorithm, state="readonly")
        algo_menu["values"] = ["Simulated Annealing", "Genetic Algorithm"]
        algo_menu.pack()

        tk.Button(self.root, text="Run Algorithm", command=self.run_selected_algorithm).pack(pady=15)
        self.result_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.pack(pady=5)
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def browse_file(self):
        file = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file:
            self.file_path.set(file)
# ========================================= Read Input ======================================= 
    def read_input_file(self, filename):
        with open(filename, "r") as file:
            lines = [line.strip() for line in file if line.strip()]
        N_vehicles = int(lines[0])
        vehicles_cap = {i + 1: float(lines[1 + i]) for i in range(N_vehicles)}
        N_packages = int(lines[1 + N_vehicles])
        packages = []
        for i in range(N_packages):
            x, y, weight, priority = map(float, lines[2 + N_vehicles + i].split())
            packages.append({"x": x, "y": y, "weight": weight,"priority":priority})
        return vehicles_cap, packages
    
# =========================================== GUI =======================================
    def run_selected_algorithm(self):
        path = self.file_path.get()
        if not path:
            messagebox.showerror("Error", "Please select an input file first.")
            return
        try:
            vehicles_cap, packages = self.read_input_file(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read input file:\n{e}")
            return
        if self.algorithm.get() == "Simulated Annealing":
            solution, cost = simulated_annealing(vehicles_cap, packages)
        else:
            solution, cost = genetic_algorithm(vehicles_cap, packages)
        self.result_label.config(text=f"Total Distance: {cost:.2f} km")
        self.plot_solution(solution, packages)

    def plot_solution(self, solution, packages):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'cyan', 'magenta']
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Vehicle Routes")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.plot(0, 0, marker='o', markersize=10, color='black', label='Shop (0,0)')

        for i, (vehicle_id, pkg_indices) in enumerate(solution.items()):
            color = colors[i % len(colors)]
            x_points = [0]
            y_points = [0]
            for pkg_idx in pkg_indices:
                pkg = packages[pkg_idx]
                x_points.append(pkg["x"])
                y_points.append(pkg["y"])
                ax.scatter(pkg["x"], pkg["y"], color=color, marker='x')
                ax.text(pkg["x"], pkg["y"], f'{pkg_idx}', fontsize=9)
            ax.plot(x_points, y_points, label=f'Vehicle {vehicle_id}', color=color, linewidth=2)

        ax.legend()
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def launch_gui_app():
    root = tk.Tk()
    app = DeliveryOptimizerApp(root)
    root.mainloop()
if __name__ == "__main__":
    launch_gui_app()
