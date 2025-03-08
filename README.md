# Bee Colony TSP Solver

Este proyecto implementa un algoritmo de colonia de abejas para resolver el problema del viajero de comercio (TSP) utilizando la librería [tsplib95](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) para cargar instancias del TSP. Se han optimizado las operaciones mediante NumPy para aprovechar cálculos vectorizados y se han incorporado diversos operadores de vecindad (swap e inversión) junto con un mecanismo de reinicialización para mejorar la exploración de soluciones.

## Estructura del Proyecto

```plaintext
bee
 ┣ TSPLIB
 ┃ ┣ berlin52.opt.tour
 ┃ ┣ berlin52.tsp
 ┃ ┣ ch130.opt.tour
 ┃ ┣ ch130.tsp
 ┃ ┣ pcb442.opt.tour
 ┃ ┗ pcb442.tsp
 ┣ src
 ┃ ┣ __init__.py
 ┃ ┣ beecolony.py
 ┃ ┣ main.py
 ┃ ┗ vectorized_beecolony.py 
 ┣ .gitignore
 ┣ README.md
 ┗ requirements.txt
```

- **TSPLIB/**: Contiene instancias TSP y tours óptimos en formato TSPLIB.
- **src/**:
  - `vectorized_beecolony.py`: Implementación del algoritmo con operaciones vectorizadas.
  - `main.py`: Script que carga los datos, ejecuta el algoritmo y muestra los resultados.
- **requirements.txt**: Lista de dependencias del proyecto.

## Requisitos

- Python 3.x
- Dependencias (instalables vía pip):
  - `numpy`
  - `tsplib95`
  - `matplotlib`
  - `tqdm`
  - `pandas`

Para instalar las dependencias, ejecuta:

```bash
pip install -r requirements.txt
```

## Ejecución
El proyecto se ejecuta desde la línea de comandos. Por ejemplo, para resolver la instancia berlin52, utiliza el siguiente comando:

```bash
python3 -m src.main \
  --tsp TSPLIB/berlin52.tsp --tour TSPLIB/berlin52.opt.tour \
  --n_bees 300 --active_bees 0.50 --explorer_bees 0.30 --max_epochs 2000
```

### Parámetros
- tsp: Ruta al archivo .tsp que define la instancia TSP.
- tour: Ruta al archivo .opt.tour con el tour óptimo conocido.
- n_bees: Número total de abejas en la colonia.
- active_bees: Proporción de abejas activas (por ejemplo, 0.50 significa el 50%).
- explorer_bees: Proporción de abejas exploradoras o scouts.
- max_epochs: Número máximo de iteraciones (épocas) a ejecutar.
- acceptance_prob: Probabilidad de aceptar una solución peor (por defecto 0.00).