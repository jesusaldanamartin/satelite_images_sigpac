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

## 14/10/2022

ENLACE even-odd rule : https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule 
Tiene pinta que funciona bien, queda mejorar la eficiencia utiliando los 4M de puntos 311 días.
Utilizando la máquina virtual del laboratorio, lo que ayer tardó 2 horas, ahora 15 minutos y 52 segundos. Mover .tif a local

## 17/10/2022

Numero de iteracioes:

- Leemos el shapefile puede ser muy grande el SP29000 = 47162 recintos.
- Por cada recinto leo mi lista de puntos uno a uno.
- Por cada punto compruebo si este está dentro de un recinto en concreto.
 
len(recinto) ->  varía entre 5 y 100 bordes aprox.
47162 * (3887458 * (len(recinto)) ) -> (1.833.402.941.960)

Hace demasiadas iteraciones, es inviable continuar con este método. Pero si no compruebo los puntos de su arr por cada recinto, como se que pixeles de sus MASK cambiar ¿? , además que no tienen porque coincidir los pixeles por algunas décimas y puede mover el resultado en algunos pixeles.

NUMBA ¿? -> mejora considerablemente

## 20/10/2022

Crear máscara para cada uno de los shapefiles y realizar algoritmos para cada uno. Despues mergeamos todos en un único TIFF.

Máscara hecha, ahora trabajando en final_version.py

Ejecutando con  el decorador @jit el municipio de fuengirola:

El municipio tiene un tamaño de 102472 puntos (malaga provincia tenia 4M) y tiene 99 geometrías diferentes (malaga provincia tenia 48K), en mi pc con numba va a 1,4 iteraciones/s . TARDANDO EN TOTAL = 3 minutos y 32 segundos.
                                            En maquina virtual ha tardado = 2 minutos y 26 segundos.
El municipio de fuengirola salida correcta comprobada.

Ejecutar el municipio de cartama ahora es viable pero tarda 70 horas. Cartama tiene 1,1M pixeles y 25K geometrias es aproximadamente la mitad de grande que malaga provincia.  12,5 KB tamaño de archivo.

Ejecutando el municipio de humilladero tarda aprox 1 hora, este municipio posee 346522 pixeles y 1232 geomtrias. 6,1 KB de tamaño de archivo.

## 21/10/2022

Completamente documentado el archivo python final del proyecto. Queda aún hacer una última prueba con un archivo muy grande tipo malaga o alhaurín. Voy a dejar en 2 plano la ejecución de shapefile de alhaurín el grande tarda unas 49 horas.

Cosas aún por hacer o mejorar:
- minio_connection.py para automatizar la descarga de su minio bucket.
- Dejar la función de utils.py para mergear todos los .tif en un único archivo. (poco viable)
- Multithreading forma más viable de mejorar tiempos. (DONE)
- Terminar de mejorar la overview del README.md 
- Poco a poco ir fusionando todas los 290**_sigpac.tif en un único archivo.

## 24/10/2022

Tener en una carpeta todos los shpaefiles junstos por cada shapefile obtener su respectiva máscara con la funcion masked_shp, guardamos cada mascara en una carpeta auxiliar. Y de nuevo por cada shapefile en la carpeta llamo a la función multithreading.

## 26/10/2022

Creando mascara de todos los archivos menores a 1164228 bytes. Descargando los datos y creando la mascara inicial de cadiz.

MALAGA = 30SUF + 30SUG
CADIZ = 29SPA + 29SQV + 30STF + 30STG + 30SUF
Granada =
Cordoba
Almeria
Sevilla
Huelva
Jaen

Cadiz son 45 municipios diferentes. Algunos municipos no hacen overlap con sus classification_29*** por lo que cadiz saldrá sin todas las parcelas(concretamente el sigpac SP20_REC_11016)

nohup python3 sigpac_raster_.....py > /dev/null 2>&1&