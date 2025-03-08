# main.py
import argparse
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tsplib95

from src.vectorized_beecolony import solve


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Algoritmo Bee Colony para resolver TSP usando tsplib95"
    )

    parser.add_argument("--tsp", type=str, required=True,
                        help="Ruta al archivo .tsp")
    parser.add_argument("--tour", type=str, required=True,
                        help="Ruta al archivo .opt.tour")

    parser.add_argument("--n_bees", type=int, default=100,
                        help="Número de abejas en la colonia")
    parser.add_argument("--active_bees", type=float, default=0.5,
                        help="Proporción de abejas activas")
    parser.add_argument("--explorer_bees", type=float, default=0.25,
                        help="Proporción de abejas exploradoras")
    parser.add_argument("--max_epochs", type=int, default=10000,
                        help="Número máximo de iteraciones")

    return parser.parse_args()


def validate_files(tsp_file, tour_file):
    """Verifica que los archivos de entrada existan."""
    if not os.path.exists(tsp_file):
        raise FileNotFoundError(
            f"❌ El archivo '{tsp_file}' no existe. Verifica la ruta.")
    if not os.path.exists(tour_file):
        raise FileNotFoundError(
            f"❌ El archivo '{tour_file}' no existe. Verifica la ruta.")


def load_optimal_tour(tour_file):
    """
    Carga la secuencia de nodos del tour óptimo.
    Se lee el archivo .opt.tour (se ignoran encabezados y EOF).
    """
    with open(tour_file, "r") as file:
        lines = file.readlines()
    tour = []
    recording = False
    for line in lines:
        if "TOUR_SECTION" in line:
            recording = True
            continue
        if recording:
            if "-1" in line or "EOF" in line:
                break
            tour.append(int(line.strip()))
    return tour


def compute_optimal_distance(optimal_tour, dist_matrix, index_map):
    """
    Calcula la distancia de la ruta óptima usando la matriz de distancias.
    Se asume que el tour es cíclico.
    """
    # Convertir el tour a índices (asumimos que en el .opt.tour se restó 1 si era 1-indexado)
    indices = [index_map[node] for node in optimal_tour]
    distance = np.sum(dist_matrix[indices[:-1], indices[1:]])
    distance += dist_matrix[indices[-1], indices[0]]
    return distance


def main():
    args = parse_arguments()
    validate_files(args.tsp, args.tour)

    # Cargar la instancia TSP
    problem = tsplib95.load(args.tsp)

    # Obtener y ordenar los nodos para mapearlos a índices 0...n-1
    nodes = sorted(problem.get_nodes())
    n_nodes = len(nodes)
    index_map = {node: i for i, node in enumerate(nodes)}

    # Precalcular la matriz de distancias
    dist_matrix = np.empty((n_nodes, n_nodes))
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            dist_matrix[i, j] = problem.get_weight(node_i, node_j)

    # Cargar tour óptimo y calcular la distancia óptima
    optimal_tour = load_optimal_tour(args.tour)
    optimal_distance = compute_optimal_distance(
        optimal_tour, dist_matrix, index_map)
    print(f"Distancia óptima conocida: {optimal_distance}")

    # Ejecutar el algoritmo y medir tiempo de ejecución
    start_time = time.time()
    best_distance, best_path, convergence = solve(
        dist_matrix, args.n_bees, args.max_epochs,
        active_ratio=args.active_bees, scout_ratio=args.explorer_bees
    )
    execution_time = time.time() - start_time
    error_percentage = (
        (best_distance - optimal_distance) / optimal_distance) * 100

    # Crear DataFrame con los resultados
    df = pd.DataFrame({
        "Distancia Final": [best_distance],
        "Tiempo de Ejecución (s)": [execution_time],
        "Error (%)": [error_percentage],
        "Distancia Óptima": [optimal_distance]
    })
    print("\nResultados:")
    print(df)

    # Graficar la convergencia
    plt.figure()
    plt.plot(convergence)
    plt.xlabel("Epoch")
    plt.ylabel("Distancia (Costo)")
    plt.title("Convergencia del Algoritmo Bee Colony")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
