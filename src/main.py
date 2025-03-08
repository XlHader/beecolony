# main.py
import argparse
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import tsplib95

from src.beecolony import solve


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
            # Se asume que los nodos vienen como enteros
            tour.append(int(line.strip()))
    return tour


def compute_optimal_distance(optimal_tour, problem):
    """
    Calcula la distancia de la ruta óptima usando el objeto problem.
    Se asume que el tour es cíclico (se retorna al nodo inicial).
    """
    distance = 0.0
    for i in range(len(optimal_tour) - 1):
        distance += problem.get_weight(optimal_tour[i], optimal_tour[i+1])
    distance += problem.get_weight(optimal_tour[-1], optimal_tour[0])
    return distance


def main():
    args = parse_arguments()
    validate_files(args.tsp, args.tour)

    # Cargar la instancia TSP y el tour óptimo
    problem = tsplib95.load(args.tsp)
    optimal_tour = load_optimal_tour(args.tour)
    optimal_distance = compute_optimal_distance(optimal_tour, problem)
    print(f"Distancia óptima conocida: {optimal_distance}")

    start_time = time.time()
    best_distance, best_path, convergence = solve(problem, args.n_bees, args.max_epochs,
                                                  args.active_bees, args.explorer_bees)
    execution_time = time.time() - start_time
    error_percentage = (
        (best_distance - optimal_distance) / optimal_distance) * 100

    # Crear DataFrame con pandas con los resultados
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
