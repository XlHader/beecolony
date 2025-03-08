import numpy as np


def compute_distance(paths, dist_matrix):
    """
    Calcula la distancia de cada ruta (cada fila de paths) de forma vectorizada.
    Se asume que cada ruta es cíclica (se retorna al inicio).
    """
    n_bees, n_nodes = paths.shape
    indices = np.arange(n_nodes - 1)
    d = np.sum(dist_matrix[paths[:, indices], paths[:, indices + 1]], axis=1)
    d += dist_matrix[paths[:, -1], paths[:, 0]]
    return d


def solve(dist_matrix, nb, max_epochs, active_ratio=0.5, scout_ratio=0.25):
    """
    Resuelve el TSP usando un algoritmo de colonia de abejas optimizado con NumPy.
    Se incorporan mejoras en la exploración mediante:
      - Operadores de vecindad variados (swap e inversión).
      - Reasignación dinámica de roles a partir de un contador de estancamiento.

    Parámetros:
      - dist_matrix: matriz NumPy (n_nodes x n_nodes) de distancias.
      - nb: número de abejas.
      - max_epochs: número máximo de iteraciones.
      - active_ratio: proporción de abejas activas.
      - scout_ratio: proporción de abejas exploradoras.

    Retorna:
      - best_distance: distancia de la mejor ruta encontrada.
      - best_path: la mejor ruta (arreglo NumPy).
      - convergence: lista con la evolución de la distancia mínima por época.
    """
    n_nodes = dist_matrix.shape[0]

    # Inicializar las rutas (cada abeja tiene una permutación aleatoria)
    paths = np.empty((nb, n_nodes), dtype=int)
    for i in range(nb):
        paths[i] = np.random.permutation(n_nodes)

    distances = compute_distance(paths, dist_matrix)
    best_index = np.argmin(distances)
    best_distance = distances[best_index]
    best_path = paths[best_index].copy()

    # Asignar roles: 0 = inactiva, 1 = scout, 2 = activa
    statuses = np.zeros(nb, dtype=np.int8)
    n_active = int(nb * active_ratio)
    n_scout = int(nb * scout_ratio)
    n_inactive = nb - (n_active + n_scout)
    statuses[n_inactive: n_inactive + n_scout] = 1
    statuses[n_inactive + n_scout:] = 2

    # Contadores de estancamiento para cada abeja (si no mejora, se reinicializa)
    stagnation = np.zeros(nb, dtype=int)
    stagnation_threshold = 50

    convergence = []
    acceptance_prob = 0.05  # probabilidad de aceptar un empeoramiento

    for epoch in range(max_epochs):
        convergence.append(best_distance)

        # --- Actualización para abejas activas (status == 2) ---
        active_indices = np.where(statuses == 2)[0]
        if active_indices.size > 0:
            # Elegir aleatoriamente el operador: True = swap, False = inversión
            operator_choice = np.random.random(size=active_indices.size) < 0.5

            # Operador swap
            swap_indices = active_indices[operator_choice]
            if swap_indices.size > 0:
                r1 = np.random.randint(0, n_nodes, size=swap_indices.size)
                r2 = np.random.randint(0, n_nodes, size=swap_indices.size)
                equal_mask = (r1 == r2)
                r2[equal_mask] = (r1[equal_mask] + 1) % n_nodes

                neighbor_paths_swap = paths[swap_indices].copy()
                idx = np.arange(swap_indices.size)
                neighbor_paths_swap[idx, r1], neighbor_paths_swap[idx, r2] = (
                    neighbor_paths_swap[idx, r2].copy(),
                    neighbor_paths_swap[idx, r1].copy()
                )
                neighbor_distances_swap = compute_distance(
                    neighbor_paths_swap, dist_matrix)
                current_distances_swap = distances[swap_indices]
                random_probs = np.random.random(size=swap_indices.size)
                update_mask_swap = (neighbor_distances_swap < current_distances_swap) | (
                    (neighbor_distances_swap >= current_distances_swap) & (
                        random_probs < acceptance_prob)
                )
                if update_mask_swap.any():
                    indices_to_update = swap_indices[update_mask_swap]
                    paths[indices_to_update] = neighbor_paths_swap[update_mask_swap]
                    distances[indices_to_update] = neighbor_distances_swap[update_mask_swap]
                    stagnation[indices_to_update] = 0  # se resetea el contador
                    if np.min(neighbor_distances_swap[update_mask_swap]) < best_distance:
                        best_distance = np.min(
                            neighbor_distances_swap[update_mask_swap])
                        best_path = paths[indices_to_update][np.argmin(
                            neighbor_distances_swap[update_mask_swap])].copy()

            # Operador inversión
            inv_indices = active_indices[~operator_choice]
            if inv_indices.size > 0:
                r1 = np.random.randint(0, n_nodes, size=inv_indices.size)
                r2 = np.random.randint(0, n_nodes, size=inv_indices.size)
                equal_mask = (r1 == r2)
                r2[equal_mask] = (r1[equal_mask] + 1) % n_nodes
                # Asegurar que para cada abeja se tenga r1 < r2 (si no, intercambiar)
                for i in range(inv_indices.size):
                    if r1[i] > r2[i]:
                        r1[i], r2[i] = r2[i], r1[i]
                neighbor_paths_inv = paths[inv_indices].copy()
                # Aplicar inversión en cada ruta
                for i in range(inv_indices.size):
                    neighbor_paths_inv[i, r1[i]:r2[i] +
                                       1] = neighbor_paths_inv[i, r1[i]:r2[i]+1][::-1]
                neighbor_distances_inv = compute_distance(
                    neighbor_paths_inv, dist_matrix)
                current_distances_inv = distances[inv_indices]
                random_probs = np.random.random(size=inv_indices.size)
                update_mask_inv = (neighbor_distances_inv < current_distances_inv) | (
                    (neighbor_distances_inv >= current_distances_inv) & (
                        random_probs < acceptance_prob)
                )
                if update_mask_inv.any():
                    indices_to_update = inv_indices[update_mask_inv]
                    paths[indices_to_update] = neighbor_paths_inv[update_mask_inv]
                    distances[indices_to_update] = neighbor_distances_inv[update_mask_inv]
                    stagnation[indices_to_update] = 0
                    if np.min(neighbor_distances_inv[update_mask_inv]) < best_distance:
                        best_distance = np.min(
                            neighbor_distances_inv[update_mask_inv])
                        best_path = paths[indices_to_update][np.argmin(
                            neighbor_distances_inv[update_mask_inv])].copy()

        # --- Actualización para abejas exploradoras (scout, status == 1) ---
        scout_indices = np.where(statuses == 1)[0]
        if scout_indices.size > 0:
            new_paths = np.empty((scout_indices.size, n_nodes), dtype=int)
            for i in range(scout_indices.size):
                new_paths[i] = np.random.permutation(n_nodes)
            new_distances = compute_distance(new_paths, dist_matrix)
            current_scout_distances = distances[scout_indices]
            update_mask = new_distances < current_scout_distances
            if update_mask.any():
                indices_to_update = scout_indices[update_mask]
                paths[indices_to_update] = new_paths[update_mask]
                distances[indices_to_update] = new_distances[update_mask]
                stagnation[indices_to_update] = 0
                if np.min(new_distances[update_mask]) < best_distance:
                    best_distance = np.min(new_distances[update_mask])
                    best_path = paths[indices_to_update][np.argmin(
                        new_distances[update_mask])].copy()

        # --- Actualización para abejas inactivas (status == 0) ---
        inactive_indices = np.where(statuses == 0)[0]
        if inactive_indices.size > 0:
            prob = 0.01
            update_mask_inactive = np.random.random(
                size=inactive_indices.size) < prob
            if update_mask_inactive.any():
                idx_inactive = inactive_indices[update_mask_inactive]
                r1 = np.random.randint(0, n_nodes, size=idx_inactive.size)
                r2 = np.random.randint(0, n_nodes, size=idx_inactive.size)
                equal_mask = (r1 == r2)
                r2[equal_mask] = (r1[equal_mask] + 1) % n_nodes
                neighbor_paths_inactive = paths[idx_inactive].copy()
                indices_range = np.arange(idx_inactive.size)
                neighbor_paths_inactive[indices_range, r1], neighbor_paths_inactive[indices_range, r2] = (
                    neighbor_paths_inactive[indices_range, r2].copy(),
                    neighbor_paths_inactive[indices_range, r1].copy()
                )
                neighbor_distances_inactive = compute_distance(
                    neighbor_paths_inactive, dist_matrix)
                current_distances_inactive = distances[idx_inactive]
                update_mask = neighbor_distances_inactive < current_distances_inactive
                if update_mask.any():
                    indices_to_update = idx_inactive[update_mask]
                    paths[indices_to_update] = neighbor_paths_inactive[update_mask]
                    distances[indices_to_update] = neighbor_distances_inactive[update_mask]
                    stagnation[indices_to_update] = 0
                    if np.min(neighbor_distances_inactive[update_mask]) < best_distance:
                        best_distance = np.min(
                            neighbor_distances_inactive[update_mask])
                        best_path = paths[indices_to_update][np.argmin(
                            neighbor_distances_inactive[update_mask])].copy()

        # --- Mecanismo de reinicialización por estancamiento ---
        # Se incrementa para todas las abejas que no hayan sido actualizadas (ya se reseteó donde hubo mejora)
        stagnation += 1
        reinit_indices = np.where(stagnation > stagnation_threshold)[0]
        if reinit_indices.size > 0:
            for idx in reinit_indices:
                paths[idx] = np.random.permutation(n_nodes)
                distances[idx] = compute_distance(
                    paths[idx][None, :], dist_matrix)[0]
                stagnation[idx] = 0
                statuses[idx] = 1  # se reclasifica como scout
                if distances[idx] < best_distance:
                    best_distance = distances[idx]
                    best_path = paths[idx].copy()

    return best_distance, best_path, convergence
