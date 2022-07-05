# Cooling Table Generator and Test Scripts
Author: Pedro Naethe Motta (pedronaethemotta@usp.br)

This code generates a radiative cooling table following the equations from [Esin et al. 1996](https://ui.adsabs.harvard.edu/abs/1996ApJ...465..312E). The radiative cooling table depends on the parameters $R$, $T_e$, $n_e$ (radius, electron temperature and electron number density) in CGS, using texture memory in CUDA.

## Running the code and generating the cooling table

1. Compile parameters_generator.c and run in order to generate three .txt files containing values for $R$, $n_e$ and $T_e$. Standard code provides you 100 values 
for each.

**note: Please, becareful, if you wish to change the number of values for each parameters, for example, from 100 to 200, you need to change the other .c files as well because they were made for 100 values.**

2. Compile cooling_table.c and run in order to generate a .txt file containing a table $(100 \times 100 \times 100)$ with parameters $R$, $n_e$ and $T_e$ and cooling values. The first line of the file indicates what each column represents.

3. Compile cooling_texture.cu and run in order to use your GPU to grab this values from a texture memory location. This will help to introduce a faster 
cooling in GPU based GRMHD simulations. 

## Testing Texture Memory

It is important to test whether texture memory is fetching the right values for different combination of the parameters. To do so you'll need to modify the ```cooling_function()``` variables v1, v2 and v3, which represent $R$, $n_e$, $T_e$ parameters, respectively.

**note: If you modified the number of values for each parameter, you'll need to adjust $N_x$, $N_y$ and $N_z$ for each function inside this file.**


