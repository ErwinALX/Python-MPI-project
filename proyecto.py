#!/usr/bin/env python
import random
import numpy
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
   R=int(input('Digite el valor de R'))
   S=float(input('Digite el valor de S'))
   defaultS=0.8
   # Hacemos la transformacion de xml a Objeto aqui 
   

else:
    data = None

print 'rank',rank,'has data:',data
comm.Barrier()
# luego buscamos los vuelos con tokens trip id diferentes y sacamos todos sus datos aqui  
   # luego con esos datos normalizamos la medicion del acelerometro 
   # (no tengo idea)luego hay que hacer la busqueda de los valores cercanos a R e ir decrementando ese con el metodo de ventana corrediza 
   # luego nose 
