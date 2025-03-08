# beecolony.py
from enum import Enum
import random
import sys


def tour_distance(path, problem):
    """
    Calcula la distancia total de una ruta 'path' usando el objeto 'problem' de tsplib95.
    Se asume que la ruta es cíclica (se vuelve al punto de inicio).
    """
    distance = 0.0
    for i in range(len(path) - 1):
        distance += problem.get_weight(path[i], path[i+1])
    distance += problem.get_weight(path[-1], path[0])
    return distance


class Status(Enum):
    inactive = 0
    active = 1
    scout = 2


class Bee(object):
    def __init__(self, nodes, problem):
        self.status = Status.inactive
        # Se parte de una copia de la lista de nodos y se mezcla aleatoriamente
        self.path = nodes[:]
        random.shuffle(self.path)
        self.distance = tour_distance(self.path, problem)


def solve(problem, nb, max_epochs, active_ratio=0.5, scout_ratio=0.25):
    """
    Resuelve el TSP usando el algoritmo de colonia de abejas.

    Parámetros:
      - problem: instancia del TSP cargada con tsplib95.
      - nb: número de abejas en la colonia.
      - max_epochs: número máximo de iteraciones (epochs).
      - active_ratio: proporción de abejas activas.
      - scout_ratio: proporción de abejas exploradoras.

    Retorna:
      - best_distance: la distancia de la mejor ruta encontrada.
      - best_path: la mejor ruta encontrada.
      - convergence: lista con la evolución de la distancia mínima por epoch.
    """
    nodes = list(problem.get_nodes())
    hive = [Bee(nodes, problem) for _ in range(nb)]

    best_distance = sys.float_info.max
    best_path = None
    for bee in hive:
        if bee.distance < best_distance:
            best_distance = bee.distance
            best_path = bee.path[:]

    num_active = int(nb * active_ratio)
    num_scout = int(nb * scout_ratio)
    num_inactive = nb - (num_active + num_scout)

    # Asignar estados a las abejas
    for i, bee in enumerate(hive):
        if i < num_inactive:
            bee.status = Status.inactive
        elif i < num_inactive + num_scout:
            bee.status = Status.scout
        else:
            bee.status = Status.active

    convergence = []  # Para registrar la mejor distancia en cada epoch
    epoch = 0
    while epoch < max_epochs:
        convergence.append(best_distance)
        if best_distance == 0.0:
            break

        for bee in hive:
            if bee.status == Status.active:
                # Genera un vecino mediante intercambio (swap) de dos nodos
                neighbor_path = bee.path[:]
                i1 = random.randint(0, len(neighbor_path) - 1)
                i2 = random.randint(0, len(neighbor_path) - 1)
                if i1 == i2:
                    i2 = (i1 + 1) % len(neighbor_path)
                neighbor_path[i1], neighbor_path[i2] = neighbor_path[i2], neighbor_path[i1]
                neighbor_distance = tour_distance(neighbor_path, problem)

                p = random.random()
                if neighbor_distance < bee.distance or (neighbor_distance >= bee.distance and p < 0.05):
                    bee.path = neighbor_path
                    bee.distance = neighbor_distance
                    if bee.distance < best_distance:
                        best_distance = bee.distance
                        best_path = bee.path[:]
                        print(
                            f"Epoch {epoch}: nueva mejor ruta con distancia = {best_distance}")

            elif bee.status == Status.scout:
                # La abeja exploradora genera un camino aleatorio
                random_path = nodes[:]
                random.shuffle(random_path)
                random_distance = tour_distance(random_path, problem)
                if random_distance < bee.distance:
                    bee.path = random_path
                    bee.distance = random_distance
                    if bee.distance < best_distance:
                        best_distance = bee.distance
                        best_path = bee.path[:]
                        print(
                            f"Epoch {epoch}: nueva mejor ruta con distancia = {best_distance}")
            # Las abejas inactivas no realizan acción
        epoch += 1

    print("\nMejor ruta encontrada:")
    print(" -> ".join(str(x) for x in best_path))
    print(f"\nDistancia de la mejor ruta = {best_distance}")
    return best_distance, best_path, convergence
