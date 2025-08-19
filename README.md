# 📦 Package Delivery Optimization

## 📖 Overview

This project implements an **AI-based optimization system** for a local package delivery shop.
It assigns packages to vehicles and determines optimal delivery routes while minimizing total traveled distance and respecting vehicle capacity limits.

The project uses two **metaheuristic algorithms**:

* 🧊 **Simulated Annealing (SA)**
* 🧬 **Genetic Algorithm (GA)**

A **Tkinter GUI** is included, allowing users to:

* Load an input file containing vehicles and package data.
* Choose which algorithm to run.
* Visualize delivery routes on a 2D map using Matplotlib.

---

## 🚀 Features

* Supports multiple vehicles with capacity constraints.
* Handles package weights, priorities, and coordinates.
* Implements **Simulated Annealing** and **Genetic Algorithm** from scratch.
* Interactive GUI to run experiments and visualize results.
* Displays total distance traveled by all vehicles.

---

## 📂 Project Structure

```
├── Project1_AI.py              # Main source code with GUI + algorithms
├── Project_AI_Report.pdf       # Project report with test cases and results
├── ENCS3340_Project_doc.pdf    # Original project description
└── README.md                   # Project documentation
```

---

## 📥 Installation

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/PackageDeliveryOptimizer.git
cd PackageDeliveryOptimizer
```

### 2. Install Dependencies

Make sure you have **Python 3.8+** installed. Then install the required libraries:

```bash
pip install matplotlib
```

> Tkinter is included by default in most Python distributions.

---

## 📌 Usage

### 1. Prepare Input File

Create a `.txt` input file with the following format:

```
<number_of_vehicles>
<capacity_vehicle1>
<capacity_vehicle2>
...
<number_of_packages>
<x y weight priority>
<x y weight priority>
...
```

✅ Example:

```
2
100
100
4
10 10 30 1
20 20 40 2
30 30 50 3
40 40 60 4
```

### 2. Run the Application

```bash
python Project1_AI.py
```

### 3. Use the GUI

* Browse and load the input file.
* Select **Simulated Annealing** or **Genetic Algorithm**.
* Click **Run Algorithm** to view:

  * Total delivery distance.
  * Optimized vehicle routes plotted on a map.

---

## 🧪 Example Results

Some experiments showed:

* **GA** performs better for large/complex inputs.
* **SA** often fine-tunes solutions better in smaller/medium cases.
* Both algorithms successfully respect vehicle capacities and reject impossible inputs.

---

## 📊 Algorithms Implemented

### 🔹 Simulated Annealing (SA)

* Starts with a random solution.
* Iteratively makes small changes (neighbors).
* Accepts worse solutions with a probability that decreases as temperature cools.

### 🔹 Genetic Algorithm (GA)

* Starts with a random population of solutions.
* Uses **selection, crossover, mutation** to evolve solutions.
* Retains the best-found solution over generations.

---

## 📖 Documentation

For detailed explanations, test cases, and results:
👉 See **[Project\_AI\_Report.pdf](./Project_AI_Report.pdf)**

For original project requirements:
👉 See **[ENCS3340\_Project\_doc.pdf](./ENCS3340_Project_doc.pdf)**

---

## 👨‍💻 Author

**Yahya Sarhan**

* Birzeit University – Computer Engineering
* Course: ENCS3340 (Artificial Intelligence)
* Instructor: Dr. Yazan Abu Farha

