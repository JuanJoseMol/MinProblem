import jax.numpy as np 

def meshgrid_from_subdiv(shape, lims = (-1,1)):
  """
  Crea una malla de N-cubo [-1,1]^N, donce N=len(shape) (tamaño de la dupla)

  # Argumentos 
  - `shape`: n-tupla, numero de puntos en cada dimension.
  - `lims`: 2-tupla, limite inferior y superior, el mismo en cada dimension. 
    
  # Resultado
  - `grid`: una grilla de n+1 dimensiones, con `grid.shape[:-1] = shape` y
    `grid.shape[-1] = n`.
  """
  liml, limr = lims
  coords = []
  for n in shape:
    coords.append(np.linspace(liml, limr, n))
  grid = np.stack(np.meshgrid(*coords, indexing='ij'), axis = -1)
  return grid 

def flatten_all_but_lastdim(array): 
  """
  Colapsa todas las dimensiones en 1, salvo la última. 
  # Ejemplo 
  ```
  >>> a = np.ones((3,5,2,6)) # a.shape = (3,5,2,6)
  >>> flatten_all_but_lastdim(a).shape
  (30, 6)
  ```  
  Notar que `30 = 3 * 5 * 2`.
  """
  return np.reshape(array, (-1, array.shape[-1]))