## 28/09/22

Read classification_30SUF y su nparray es de 10980x10980 pixeles con formato:

[[0 0 0 0 ... 0 0 0 0]
 [0 0 0 0 ... 0 0 0 0]
 [0 0 0 0 ... 0 0 0 0]
 [0 0 0 0 ... 0 0 0 0]]

 Si cambio mi ndarray con linspace a ese formato y actualizo la banda de su tiff, filtrando por mi shape¿?

 ## 30/09/2022

 Creo narray con width y length del municipio e inserto 0 en caso de no pertenecer a la parcela o 1-30 de identificador de uso sigpac.

Recorro shpefilr obtengo cod_uso para cada geometry, cada geometry obtengo su linspace para x e y, 

si guardo en una lista el tamaño de cada x e y ,

puedo crear un np.zeros del tamaño resultante, y tengo los cod_uso de cada parcela pero como reemplazo en la posicion correcta del np por cada codigo

[[...]
[...]
[...]
[...]]

mask of classification_suf, nested loop for each band != 0 its dataset.transform * (i,j) , then look out which cod_uso has that pixel

### Para proximo dia

Get all i,j positions that has no zero value, and save that i,j values, then get coordinates for those i,j by dataset.transform * (i,j).
After that you got all important coordinates, now look for every coord in which polygon_geometry is contained so you can get the cod_uso of that geometry. Eventually you will need to replace in those i,j positions its cod_uso.

## 03/10/2022

Iterate trough the shapefile (find in which format are the cordinates and find if point are contained in them) if true just swap it in the correspondant arr.

1.- Cre

## 05/10/2022

Unintled-1 funciones antes de cambiar a apply.
sigue el mismo problema obtengo los indices de manera correcta pero el tiempo de iteraccion es muy largo.

## 07/10/2022

Realizando prueba con len(points_list)-3000000 , aprox unos 800000 puntos.

Encontrar forma para mejorar la eficiencia del codigo
- ind_arr = [transformer.rowcol(points_list[ind][0],points_list[ind][1]) for ind in ind_values]

Formas de mejorar la eficiencia:
- List Comprehensions, apply, vectorization
- Si por cada parcela obtengo las coordenadas que la componen acelero mucho el calculo ya que en una misma iteracion del loop(features in layer) 
- 
- Dividir la lista en cuatro porciones (da igual por donde dividir) y ejecutar la funcion inside() en simultaneo, así tiro mas de CPU y lo mismo es viable ejecutar los 3800000 de pixeles.

RUN INFORMATION:

- Nº de pixeles transformados a banda SIGPAC = 884588
- Nº of Geometries in Malaga province = 47162 (al menos 3000 geometrias que se encuentran fuera del classification_30SUF)
- Execution time: 5598.637432575226 seconds -> 1,555177 hours

## 13/10/2022

Al utilizar el core algorithm para calcular correctamente si un punto está en una geometría aumenta la complejidad del bucle una locura y añade mucho tiempo de espera, aproximadamente 2 horas para 3260 puntos. El día anterior conseguí 884588 puntos cambiados en 1,5 horas. Probando el multiprocessing falla. 

Buscar forma de dividir la lista de pixeles en 4 para ejecutarlos en paralelo y al mismo tiempo guardar el indice correcto para luego sustituir en el arr de source.

