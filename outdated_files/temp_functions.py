
st = time.time()

def inside(xmin,ymin,xmax,ymax, lista_pixels):
    index_points=[]
    # list_points = [index_points.append(i) for i in range(len(lista_pixels)) if xmin <= lista_pixels[i][0] <= xmax and ymin <= lista_pixels[i][1] <= ymax]
    for i in range(len(lista_pixels)):
        if xmin <= lista_pixels[i][0] <= xmax and ymin <= lista_pixels[i][1] <= ymax:
            index_points.append(i)
    return index_points

def pixel_index(row, coordinate1,coordinate2):
    ind = []
    split_row =  row['bounds'].split(",")
    xmin = float(split_row[0][1:])
    ymin = float(split_row[1])
    xmax = float(split_row[2])
    ymax = float(split_row[3][:-1])
    if xmin <= coordinate1 <= xmax and ymin <= coordinate2 <= ymax:
        ind.append((coordinate1,coordinate2))
        return ind
    else: 
        return np.nan

def apply_func(path,list_pixels):
    df = pd.read_csv(path)
    for elem in list_pixels:
        df2 = df.apply(pixel_index, args=(elem[0],elem[1]), axis = 1)
        print(df2.dropna())
    return 

# apply_func("C:\\TFG_resources\\satelite_images_sigpac\\bounds_cod.csv",list_pixels)

def get_band_matrix2(path):
    df = pd.DataFrame(columns=['bounds','cod_uso'])
    cont = 0
    with fiona.open(path) as layer:
        for feature in layer:
            ord_dict = feature['properties']
            for key in ord_dict.values():
                if key in cod_uso: 
                    df.loc[cont,'cod_uso'] = get_id_codigo_uso(key)
                    # print(cd_uso)
                    #* codigo sigpac para cada iteraccion del shapefile
            geometry = feature["geometry"]['coordinates']
            for g in geometry:
                polygon = Polygon(g)
                bounds = polygon.bounds   
                df.loc[cont,'bounds'] = bounds
                cont+=1
                print(cont)
    return df.to_csv("bounds_cod.csv")

# get_band_matrix2("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp")

with rasterio.open("C:\TFG_resources\satelite_images_sigpac\maskSUF_raster_29900.tif") as src:
    profile = src.profile
    arr = src.read(1)
    not_zero_indices = np.nonzero(arr) # Get all indices of non-zero values in array
    # print(np.array((not_zero_indices[0],not_zero_indices[1])))
    points_list = [src.transform * (not_zero_indices[0][i],not_zero_indices[1][i]) for i in range(len(not_zero_indices[0]))] 
    matrix = get_band_matrix2("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp",
                            points_list)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    with rasterio.open('zz_salida.tif', 'w', **profile) as dst:
        dst.write(matrix, 1)
