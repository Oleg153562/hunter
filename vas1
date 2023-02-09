#include "layer.h"
#include "cuda.h"

#include <stdlib.h>

void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if(l.rand)           free(l.rand);
#ifdef GPU
        if(l.rand_gpu)             cuda_free(l.rand_gpu);
