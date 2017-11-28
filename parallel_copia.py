from __future__ import print_function
import xml.etree.cElementTree as etree
import pickle
import bisect
import os
from math import sin, cos, sqrt, asin, trunc, radians
import argparse
from collections import namedtuple
import json
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

Rules = namedtuple('Rules', ['r', 's', 'n'])
Point = namedtuple('Point',
                   ['stamp', 'latitude', 'longitude', 'altitude', 'speed', 'accelerometer'])
SemiPoint = namedtuple('SemiPoint', ['latitude', 'longitude'])
Box = namedtuple('Box', ['inner', 'frontier'])

FILE = 'data/raw.xml'
TOTAL_ITEMS = 162650

Radio_Tierra = 6371.0
NORTH_POLE = Point(stamp=None, latitude=90.0, longitude=0.0, altitude=None, speed=None,
                   accelerometer=None)


class ArgumentParser:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('-r', help='Ingrese el radio R para calculo de medias.', type=int)
        parser.add_argument('-s', help='Ingrese el valor S (score) por defecto', type=float)
        parser.add_argument('-n', help='Ingrese el numero de nodos para la division de calculos', type=int)

        parser_namespace = parser.parse_args()

        self.rules = None

        self.validateRules(parser_namespace)

    def validateRules(self, namespace):
        final_rules = [0] * 3
        if namespace.r is None:
            final_rules[0] = 30
        else:
            final_rules[0] = namespace.r

        if namespace.s is None or namespace.s > 1 or namespace.s < 0:
            final_rules[1] = 0.8
        else:
            final_rules[1] = namespace.s

        if namespace.n is None or namespace.n < 0:
            final_rules[2] = 1
        else:
            final_rules[2] = namespace.n

        self.rules = Rules(
            r=final_rules[0],
            s=final_rules[1],
            n=size
        )

    def getRules(self):
        return self.rules


class DataLoader:
    TIMESTAMP_FILE = 'data/timestamps.dat'
    PICKLE_FILE = 'data/data_file.dat'
    RESULT_FILE = 'data/result.dat'

    def __init__(self, filename):
        self.filename = filename

    
    @staticmethod
    def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='*'):
        porcentajebarra = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, porcentajebarra, suffix), end='\r')

        # Print New Line on Complete
        if iteration == total:
            print()

    def read(self, filename):
        context = iter(etree.iterparse(filename, events=('start', 'end')))
        _, root = next(context)  

        for event, elem in context:
            if event == 'end' and elem.tag == 'table':
                yield (
                    elem[1].text, elem[2].text, elem[3].text, elem[4].text,
                    elem[5].text, elem[6].text, elem[7].text)
                root.clear()  

    def _load(self):
        route_info = dict()
        DataLoader.printProgressBar(0, TOTAL_ITEMS, prefix='Analizando archivo:', suffix='Terminado')
        counter = 0
        for i in self.read(self.filename):
            DataLoader.printProgressBar(
                counter + 1, TOTAL_ITEMS, prefix='Analizando archivo:',
                suffix='Completado (Trip: ' + i[0] + ')')

            if route_info.get(i[0]) is None:
                route_info[i[0]] = []
            point = Point(
                stamp=int(i[1]),
                latitude=float(i[2]),
                longitude=float(i[3]),
                altitude=float(i[4]),
                speed=float(i[5]),
                accelerometer=float(i[6]),
            )
            route_info[i[0]].append(point)
            counter += 1

        self.save(route_info)
        return route_info

    def load(self, lowcost=True):
        if not lowcost:
            return self._load()
        else:
            print('--> Cargando los costos menores <--')
            self.tstamps = self.loadTimeStamps()
            if self.checkTimeStamps():
                print('--> Marcas de tiempo coinciden. Cargando copia local <--')
                return self._loadPkl()
            else:
                print('--> Las marcas de tiempo no coinciden. Regresando accion <--')
                return self._load()

    def loadTimeStamps(self):
        with open(DataLoader.TIMESTAMP_FILE, 'rb') as jfile:
            tstamps = json.loads(jfile.read().decode('utf-8'))

        return tstamps

    def saveResult(self, result):
        res = [(str(avg), str(lat), str(lng)) for avg, lat, lng in result]
        res = ['  '.join(r) for r in res]
        res = '\n'.join(res)

        with open(DataLoader.RESULT_FILE, 'wb') as rfile:
            rfile.write(res.encode('utf-8'))

    def saveTimeStamps(self):
        with open(DataLoader.TIMESTAMP_FILE, 'wb') as jfile:
            jfile.write(json.dumps(self.tstamps, sort_keys=True, indent=4).encode('utf8'))

    def checkTimeStamps(self):
        real_time = int(os.path.getmtime(self.filename))
        last_time = self.tstamps['file-load']

        return real_time == last_time

    def _loadPkl(self):
        with open(DataLoader.PICKLE_FILE, 'rb') as handle:
            b = pickle.load(handle)
        print('-> Terminado <-')

        return b

    def getTimeStamps(self):
        return self.tstamps

    def savePkl(self, data):
        with open(DataLoader.PICKLE_FILE, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save(self, data):
        self.tstamps = self.loadTimeStamps()
        self.tstamps['file-load'] = int(os.path.getmtime(self.filename))

        self.saveTimeStamps()
        self.savePkl(data)


def measure(p1, p2=NORTH_POLE):
    '''
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    '''

    # Convirtiendo grados decimales a radianes
    lon1, lat1, lon2, lat2 = map(radians, [p1.longitude, p1.latitude, p2.longitude, p2.latitude])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (sin(dlat / 2))**2 + cos(lat1) * cos(lat2) * (sin(dlon / 2)**2)
    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers is 6371
    meters = 1000 * Radio_Tierra * c
    return meters


def findTraveledDistance(trip):
    distance = 0

    # Ordena los puntos por marcas de tiempo
    trip.sort(key=lambda x: x.stamp)

    for i in range(0, len(trip), 2):
        if i == len(trip) - 1:
            continue
        distance += measure(trip[i], trip[i + 1])

    return distance


def measurePlayground(trips):
    min_lat, min_lng = 999, 999
    max_lat, max_lng = -999, -999

    for trip in trips:
        for point in trips[trip]:
            if max_lat < point.latitude:
                max_lat = point.latitude
            if min_lat > point.latitude:
                min_lat = point.latitude
            if max_lng < point.longitude:
                max_lng = point.longitude
            if min_lng > point.longitude:
                min_lng = point.longitude

    return (max_lat, min_lat, max_lng, min_lng)


def slicePlayground(trips, limits, radius, nodes):
    # box_dimensions = 2 * radius
    # frontier_width = radius

    ref_lat, ref_lng = limits[1], limits[3]
    ref_pnt = SemiPoint(latitude=limits[1], longitude=limits[3])
    ref_pnt_x = SemiPoint(latitude=limits[1], longitude=limits[2])
    ref_pnt_y = SemiPoint(latitude=limits[0], longitude=limits[3])

    # Ubica la posicion de cada punto en los viajes
    positions = dict()

    playground_x = measure(ref_pnt, ref_pnt_x)
    playground_y = measure(ref_pnt, ref_pnt_y)

    for trip in trips:
        positions[trip] = [0] * len(trips[trip])
        for i in range(len(trips[trip])):
            lat_pos = SemiPoint(latitude=trips[trip][i].latitude, longitude=ref_lng)
            lng_pos = SemiPoint(latitude=ref_lat, longitude=trips[trip][i].longitude)

            positions[trip][i] = (measure(ref_pnt, lng_pos), measure(ref_pnt, lat_pos))

    # Get slices, working in meters from now on
    slices_x = [playground_x]
    slices_y = [playground_y]

    print('Area construida', int(playground_x), 'x', int(playground_y))
    while len(slices_x) * len(slices_y) < nodes:
        if len(slices_x) > len(slices_y):
            new_size = len(slices_y) + 1
            slices_y = [playground_y * (i + 1) / new_size for i in range(new_size)]
        else:
            new_size = len(slices_x) + 1
            slices_x = [playground_x * (i + 1) / new_size for i in range(new_size)]

    node = 'nodes' if rules.n > 1 else 'node'
    area = 'region' if len(slices_x) * len(slices_y) > 1 else 'regions'
    print('Ajustando', len(slices_x) * len(slices_y), area, 'en', nodes, node)

    # Allocate the points to their respective areas
    areas_points = [[dict() for i in range(len(slices_y))] for i in slices_x]

    for trip in positions:
        for i in range(len(positions[trip])):
            x = trunc(positions[trip][i][0] / ((playground_x + 1) / len(slices_x)))
            y = trunc(positions[trip][i][1] / ((playground_y + 1) / len(slices_y)))

            if areas_points[x][y].get(trip) is None:
                areas_points[x][y][trip] = list()

            areas_points[x][y][trip].append(trips[trip][i])

    # Flatten areas
    points = [
        areas_points[i][j]
        for i in range(len(areas_points)) for j in range(len(areas_points[i]))]

    # Gather excess in last area
    if len(points) > nodes:
        check = True
        while len(points) > nodes and check:
            elements = countAreasFlatten(points)
            if 0 not in elements:
                check = False
            del points[elements.index(0)]

    while len(points) > nodes:
        elements = countAreasFlatten(points)
        min_idx = elements.index(min(elements))
        del elements[min_idx]

        elem = points.pop(min_idx)
        min_idx = elements.index(min(elements))

        for trip in points[min_idx]:
            for point in elem[trip]:
                points[min_idx][trip].append(point)

    return points


def countAreas(areas_points):
    for i in range(len(areas_points)):
        for j in range(len(areas_points[i])):
            acum = 0
            for trip in areas_points[i][j]:
                for point in areas_points[i][j][trip]:
                    acum += 1
            print(acum)


def countAreasFlatten(areas_points):
    acums = list()
    for i in areas_points:
        acum = 0
        for trip in i:
            for point in i[trip]:
                acum += 1
        acums.append(acum)
    return acums


def maxAcc(trip, trips):
    maxi = -9999
    for i in trips[trip]:
        if maxi < i.accelerometer:
            maxi = i.accelerometer

    return maxi


def minAcc(trip, trips):
    mini = 9999
    for i in trips[trip]:
        if mini > i.accelerometer:
            mini = i.accelerometer

    return mini


def maxAccFlat(trip, trips):
    maxi = -9999
    for i in trips:
        if i[0] == trip:
            if maxi < i[1].accelerometer:
                maxi = i[1].accelerometer

    return maxi


def minAccFlat(trip, trips):
    mini = 9999
    for i in trips:
        if i[0] == trip:
            if mini > i[1].accelerometer:
                mini = i[1].accelerometer

    return mini


def sanitizeData(trips):
    # Normalizando mediciones de acelerometro
    acelerometro = dict()

    for trip in trips:
        for point in trips[trip]:
            if acelerometro.get(trip) is None:
                acelerometro[trip] = [9999, -9999]
            if acelerometro[trip][0] > abs(point.accelerometer):
                acelerometro[trip][0] = abs(point.accelerometer)
            if acelerometro[trip][1] < abs(point.accelerometer):
                acelerometro[trip][1] = abs(point.accelerometer)

    for trip in trips:
        for i in range(len(trips[trip])):
            new_acc = (trips[trip][i].accelerometer - aceletrometro[trip][0]) / (acelerometro[trip][1] - acc[trip][0])
            trips[trip][i] = Point(
                stamp=trips[trip][i].stamp,
                latitude=trips[trip][i].latitude,
                longitude=trips[trip][i].longitude,
                altitude=trips[trip][i].altitude,
                speed=trips[trip][i].speed,
                accelerometer=new_acc
            )

    return trips


def getAreaStatus(slices):
    def multiplier(max_stamp, min_stamp, curr_stamp):
        max_mult = 2
        min_mult = 1

        prev_range = max_stamp - min_stamp
        new_range = max_mult - min_mult

        if prev_range == 0:
            return min_mult

        return (((curr_stamp - min_stamp) * new_range) / prev_range) + min_mult

    # Flatten trip structure to a form (trip, point)
    trips, r, s = slices

    flat_trips = list()
    enumerator = 0
    for trip in trips:
        for point in trips[trip]:
            flat_trips.append((enumerator, trip, point))
            enumerator += 1

    # Sort trips
    flat_trips.sort(key=lambda x: x[2].accelerometer)

    flat_acc = [f[2].accelerometer for f in flat_trips]
    ignore_point = bisect.bisect_right(flat_acc, s)

    considered_trips = flat_trips[ignore_point: len(flat_trips)]

    # Take radius of point and promediate
    averages = list()
    while len(considered_trips) > 0:
        reference = considered_trips.pop()

        points_in_radius = list()
        for i in range(len(flat_trips)):
            if measure(reference[2], flat_trips[i][2]) <= r:
                points_in_radius.append(flat_trips[i])

        idx_rm = list()
        for i in range(len(points_in_radius)):
            if points_in_radius[i][1] == reference[1]:
                idx_rm.append(points_in_radius[i][0])

        to_eliminate = [p for p in points_in_radius if p[0] in idx_rm]
        points_in_radius = [p for p in points_in_radius if p[0] not in idx_rm]

        # Determine time window
        min_time, max_time = 9991431528117906, 0
        for point in points_in_radius:
            if min_time > point[2].stamp:
                min_time = point[2].stamp
            if max_time < point[2].stamp:
                max_time = point[2].stamp

        # Time window average
        num = reference[2].accelerometer * multiplier(max_time, min_time, reference[2].stamp)
        den = multiplier(max_time, min_time, reference[2].stamp)
        for point in points_in_radius:
            num += point[2].accelerometer * multiplier(max_time, min_time, point[2].stamp)
            den += multiplier(max_time, min_time, point[2].stamp)

        avg = num / den
        averages.append((avg, reference[2].latitude, reference[2].longitude))

        enums = [i[0] for i in points_in_radius]
        considered_trips = [c for c in considered_trips if c[0] not in enums]
        considered_trips = [c for c in considered_trips if c[0] not in to_eliminate]

    return averages


if rank == 0:
    rules = ArgumentParser().getRules()
    loader = DataLoader(FILE)
    route_info = loader.load(lowcost=True)

    # Busca distancias de viaje si no han sido aun calculadas
    print('--> Encontrando distancia para viajes faltantes por calcular <-- ')
    count = False
    for trip in route_info:
        if loader.getTimeStamps().get(trip) is None:
            print(trip, end=' ')
            count = True
            distance = findTraveledDistance(route_info[trip])
            loader.getTimeStamps()[trip] = distance
    if count:
        print('')
    loader.saveTimeStamps()

    # Sanitation
    print('--> Limpiando datos <--')
    route_info = sanitizeData(route_info)

    # Find playground limits
    print('--> Buscando limites del area construida <--')
    limits = measurePlayground(route_info)

    nodes = 'nodes...' if rules.n > 1 else 'node...'
    print('--> Dividiendo area construida en ', rules.n, nodes)

    slices = slicePlayground(route_info, limits, rules.r, rules.n)

    print('--> Iniciando', rules.n, 'trabajos <--')

    # required info is slice, r, and n
    slices = [(s, rules.r, rules.s) for s in slices]
else:
    slices = None

slices = comm.scatter(slices, root=0)

area_status = getAreaStatus(slices)

res = comm.gather(area_status, root=0)

if rank == 0:
    print('--> Guardando resultados <--')
    rres = list()
    for r in res:
        rres += r
    loader.saveResult(rres)

    print('--> Proceso terminado <--')
