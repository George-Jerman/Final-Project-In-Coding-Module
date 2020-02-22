//
//  main.c
//  Final_Project
//
//  Created by George Jerman on 14/01/2020.
//  Copyright © 2020 George Jerman. All rights reserved.
//
/*
 ------------------------Manual---------------------------------
 This program simulates the Ising model of ferromsgnetism.
 The program takes several required command line arguments.
 These are:
 -d the dimension of the array to be used
 -j the spin spin exchange interaction (for iron J/K_b = 17.4)
 -s the inital temperature of the system
 -e the final temperature of the system
 -r the number of steps in temperature
 -b the magnetic field strength
 -i the nunber of iterations to be ran
 -f the base filename for the final array to be printed to
 -i the base filename for the initial array to be printed to
 
 sample command that I've been using is:
./a.out -f initial_array.txt -o final_array.txt -b 0 -d 10  -j 1 -s 1 -e 3 -r 20 -i 10000
 
 J = (ln(1 + sqrt(2)) * K_b * T_c)/2 all in SI units. For Tc = 1000 this gives J = 6.08147775e-21
 
 The files that are written by the program are as follows:
 intial_array.txt which holds the randomly generated array at the beginning of the simulaiton
 final_array.txt which is the array of spins once the simulation has completed
 energy_fluctuations.txt which shows the change in total energy between every ten steps
 
 The progran also will print out the current iteration to the console as well as a runtime at the end of the program.
 
 */



/*---------includes-------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h>

/*---------definitions-----------*/
#define BASE_10 10
#define SAMPLE_RANGE 10
#define MAX_FILE_NAME_LENGTH 100
#define MU_0 1.25663706212e-6 /*units of H/m*/
//#define K_B 1.380649e-23 /*units of J/K*/
#define K_B 1
#define xfree(mem) { free( mem ); mem = NULL; } //checks if free is successful
#define S(i,j) (spin[((Quantities->array_size+(i))%Quantities->array_size)*Quantities->array_size+((Quantities->array_size+(j))%Quantities->array_size)])/* Makes referring to the array simpler and also deals
                                    with the periodic BCs required to simulate an infinite lattice
                                    i.e this means that the N * N array element wraps around to the 0th
                                    element*/
#define H(i,j) (has_been_checked[((Quantities->array_size+(i))%Quantities->array_size)*Quantities->array_size+((Quantities->array_size+(j))%Quantities->array_size)])

typedef enum {
    NO_ERROR = 0,
    NO_MEMORY = 1,
    BAD_ARGS = 2,
    BAD_FILENAME = 3,
    BAD_FORMAT = 4,
    BAD_FILE_OPEN = 5,
    UNKNOWN_ERROR = -1,
    BAD_MALLOC = 100
} Error;

//put the user specifed constants as well as energy in this struct to minimise
//number of variables to pass to the functions
typedef struct Quantities {
    double J;
    double B;
    int start_temp;
    int final_temp;
    int number_of_steps;
    double temperature;
    long double total_energy;
    double magnetisation;
    double delta_e;
    unsigned long array_size;
    long number_of_iterations;
    double current_avg_energy;
    double current_avg_energy_squared;
    double current_avg_magnetisation;
    double current_avg_magnetisation_squared;
    long double current_avg_heat_cap;
    long double current_avg_susceptibility;
    long double heat_capacity;
    long double big_heat_cap;
    long double susceptibility;
    double avg_energy[SAMPLE_RANGE];
    double avg_energy_squared[SAMPLE_RANGE];
    double avg_magnetisation[SAMPLE_RANGE];
    double avg_magnetistion_squared[SAMPLE_RANGE];
    double avg_heat_capacity[SAMPLE_RANGE];
    double avg_susceptibility[SAMPLE_RANGE];
    double avg[SAMPLE_RANGE];
} quantities;


/*-----------prototypes-----------*/
static void * xmalloc(size_t bytes);
static Error get_double_arg( double *value, const char *opt_name, char *optarg);
static int checked_strtoi(const char *string);
static long checked_strtol(const char *string);
static Error populate_array(int *spin, quantities * Quantities, gsl_rng *r, char * initial_array_filename);
static double calculate_energy (int *spin, quantities * Quantities);
static double calculate_energy_change (int *spin, quantities * Quantities, unsigned long i, unsigned long j);
static Error metropolis_implementation (int *spin, quantities * Quantities, gsl_rng * rng_struct, FILE * heatcaptest);
static Error calculate_average_energy_and_heat_cap (quantities * Quantities, int rolling_count, int count, int big_count, FILE * heatcaptest);
static Error calculate_heat_capacity(quantities * Quantities);
static double calculate_magnetisation(int *spin, quantities * Quantities);
static Error calculate_average_magnetisation_and_susceptibility(quantities * Quantities, int rolling_count, int count);
static Error calculate_susceptibility(quantities * Quantities);
static Error print_final_vals(quantities * Quantities);
static Error create_and_fill_final_array(quantities * Quantities,int * spin, char * final_file_name);

int main(int argc, char ** argv) {

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);


    static struct option long_options[] = {
    /* These options don’t set a flag the are edistinguished by their indices. */
    {"matrix_size", required_argument,  0, 'd'},
    {"J", required_argument, 0, 'j'},
    {"B", required_argument, 0, 'b'},
    {"Initial_Temp", required_argument, 0, 's'},
    {"Final_Temperature", required_argument, 0, 'e'},
    {"number_of_temps", required_argument, 0, 'r'},
    {"Iterations", required_argument, 0, 'i'},
    {"Initial_array", required_argument, 0, 'f'},
    {"Final_array", required_argument, 0, 'o'},
    {0, 0, 0, 0}
    };
    //tells the program to use the mersenne twister rng algorithm.
    gsl_rng * random_num_gen_struct = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(random_num_gen_struct, clock()); // sets the seed to the clock meaning the random numbers will be different every time the code is run

    
    //creating the strucutre and setting all values initially to = 0
    quantities * Quantities = xmalloc(sizeof(quantities));
    Quantities->J=0;
    Quantities->B=0;
    Quantities->start_temp=0;
    Quantities->final_temp=0;
    Quantities->temperature=0;
    Quantities->number_of_steps=0;
    Quantities->total_energy=0;
    Quantities->magnetisation=0;
    Quantities->delta_e=0;
    Quantities->array_size=0;
    Quantities->number_of_iterations=0;
    Quantities->heat_capacity=0;
    Quantities->big_heat_cap=0;
    Quantities->susceptibility=0;
    memset(Quantities->avg_energy, 0, SAMPLE_RANGE*sizeof(double));
    memset(Quantities->avg_energy_squared, 0, SAMPLE_RANGE*sizeof(double));
    memset(Quantities->avg_magnetisation, 0, SAMPLE_RANGE*sizeof(double));
    memset(Quantities->avg_magnetistion_squared, 0, SAMPLE_RANGE*sizeof(double));
    memset(Quantities->avg_heat_capacity, 0, SAMPLE_RANGE*sizeof(double));
    memset(Quantities->avg_susceptibility, 0, SAMPLE_RANGE*sizeof(double));
    memset(Quantities->avg, 0, SAMPLE_RANGE*sizeof(double));
    
    
    char * initial_array_file_name = NULL;
    char * final_array_file_name = NULL;
    
    int ret_val = 0;
    int option_index = 0;
    int c = getopt_long( argc, argv, ":d:j:b:s:e:r:i:f:o:", long_options, &option_index );
    /* End of options is signalled with '-1' */
    while (c != -1) {
        switch (c) {
            case 'd':
                //printf("v\n");
                Quantities->array_size = checked_strtoi(optarg);
                break;
            case 'j':
                ret_val = get_double_arg(&Quantities->J, long_options[option_index].name, optarg);
                break;
            case 'b':
                ret_val = get_double_arg(&Quantities->B, long_options[option_index].name, optarg);
                break;
            case 's':
                Quantities->start_temp = checked_strtoi(optarg);
                Quantities->temperature = Quantities->start_temp;
                break;
            case 'e':
                Quantities->final_temp = checked_strtoi(optarg);
                break;
            case 'r':
                Quantities->number_of_steps = checked_strtoi(optarg);
                break;
            case 'i':
                Quantities->number_of_iterations = checked_strtol(optarg);
                break;
            case 'f':
                initial_array_file_name = optarg;
                break;
            case 'o':
                final_array_file_name = optarg;
                break;
            case ':':
                /* missing option argument */
                fprintf(stderr, "Error: option '-%c' requires an argument\n", optopt);
                return BAD_ARGS;
            case '?':
                default:
                /* invalid option */
                fprintf(stderr, "Warning: option '-%c' is invalid: ignored\n", optopt);
                break;
        }
        c = getopt_long( argc, argv, ":d:j:b:s:e:r:i:f:o:", long_options, &option_index );
    }
    double temp_range = Quantities->final_temp-Quantities->start_temp;
    double temp_change = temp_range/Quantities->number_of_steps;
    
    FILE * heatcaptest = fopen("end_data.txt", "w");
    fprintf(heatcaptest, "# C\tT\n");
    
    for (int i=0; i<Quantities->number_of_steps; i++) {
        printf("%d\n",i);
        Quantities->temperature += temp_change;
        
        int *spin = xmalloc(sizeof(int)*Quantities->array_size*Quantities->array_size);
        populate_array(spin, Quantities, random_num_gen_struct, initial_array_file_name);
        Quantities->total_energy = calculate_energy(spin, Quantities);
        metropolis_implementation(spin, Quantities, random_num_gen_struct, heatcaptest);
        create_and_fill_final_array(Quantities, spin, final_array_file_name);

        print_final_vals(Quantities);
        xfree(spin);
    }
    fclose(heatcaptest);
    gsl_rng_free(random_num_gen_struct);
    xfree(Quantities);
    

    //gets and prints the total runtime
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    unsigned long long int delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("%llu\n", delta_us);
    return 0;
}

static void * xmalloc(size_t bytes){
    void * retval = malloc(bytes);
    if (retval) {
        return retval;
    }
    printf("memory was not successfully allocated\n");
    exit(BAD_MALLOC);
}

static Error get_double_arg( double *value, const char *opt_name, char *optarg) {
    char * endptr = NULL;
    *value = strtod(optarg, &endptr);
    if (*endptr) {
        printf ("Error: option -%s has an invalid argument `%s'.\n", opt_name, optarg);
        return BAD_ARGS;
    }
    return NO_ERROR;
}

//ensures the value taken by strtoul can be processed as an integer
static int checked_strtoi(const char *string){
    char * endptr;
    errno = 0;
    unsigned long temp_long = strtoul(string, &endptr, BASE_10);
    if (errno == ERANGE || *endptr != '\0' || string == endptr) {
        printf("input dimension size is not an integer\n");
        exit(BAD_ARGS);
    }
    return (int) temp_long;

}

static long checked_strtol(const char *string){
    char * endptr;
    errno = 0;
    unsigned long temp_long = strtoul(string, &endptr, BASE_10);
    if (errno == ERANGE || *endptr != '\0' || string == endptr) {
        printf("input dimension size is not an integer\n");
        exit(BAD_ARGS);
    }
    return (long) temp_long;

}

//randomly populates the array using the Mersenne Twister rng
static Error populate_array(int *spin,  quantities * Quantities, gsl_rng *random_num_gen_struct, char * initial_array_filename){
    FILE * initial =NULL;
    static char initial_name[MAX_FILE_NAME_LENGTH];
    sprintf(initial_name,"T=%lg_%s",Quantities->temperature, initial_array_filename);
    initial = fopen(initial_name, "w");
    
    if (initial == NULL){
        printf("File failed to open\n");
        exit(BAD_FILE_OPEN);
    }
    double temp =0;
    for (int i=0; i<Quantities->array_size; i++) {
        for (int j=0; j<Quantities->array_size; j++) {
            temp = gsl_rng_uniform(random_num_gen_struct);
            //printf("%f\t", temp);
            if (temp >= 0.5) {
                S(i,j) = 1;
            }
            else{
                S(i,j) = -1;
            }
            fprintf(initial, "%d\t", S(i,j));
        }
        fprintf(initial,"\n");
    }
    fclose(initial);
    return NO_ERROR;
}

static double calculate_energy (int *spin, quantities * Quantities){
    double Energy = 0;
    for (int i=0; i<Quantities->array_size; i++) {
        for (int j=0; j<Quantities->array_size; j++) {
            Energy += -(S(i,j)*( Quantities->J*(S(i+1,j)+S(i,j+1)) + MU_0*Quantities->B ));
        }
    }
    return Energy;
}

static double calculate_energy_change (int *spin, quantities * Quantities, unsigned long i, unsigned long j){
    double delta_energy =  S(i,j)*( 2.0*Quantities->J*(S(i-1,j)+S(i,j-1)+S(i+1,j)+S(i,j+1)) + MU_0*Quantities->B );
    return delta_energy;
}

//This function implements the metropolis algorithm to evolve the system and also is responsible at present for
// calculating the energy fluctuations at ten step intervals
static Error metropolis_implementation (int *spin/*,int *has_been_checked*/, quantities * Quantities, gsl_rng * rng_struct, FILE * heatcaptest){
    unsigned long i=0, j=0;
    int rolling_count =0;
    int count =0;
    int big_count=0;
    long double new_energy_val=0;
    long double energy_change=0;
    double random=0;
    int array_counter =0;
    int *has_been_checked = xmalloc(sizeof(int)*Quantities->array_size*Quantities->array_size);
    memset(has_been_checked, 0, sizeof(int)*Quantities->array_size*Quantities->array_size);

    for (i=0; i<Quantities->array_size; i++) {
        for (j=0; j<Quantities->array_size; j++) {
            H(i, j) = 0;
        }
    }
    while (count<Quantities->number_of_iterations+1) {

        array_counter = 0;
        memset(has_been_checked, 0, sizeof(bool)*Quantities->array_size*Quantities->array_size);
        Quantities->total_energy = calculate_energy(spin, Quantities);
        Quantities->magnetisation = calculate_magnetisation(spin, Quantities);
        calculate_average_energy_and_heat_cap(Quantities, rolling_count, count, big_count,heatcaptest);
        calculate_average_magnetisation_and_susceptibility(Quantities, rolling_count, count);
        rolling_count++;
        big_count++;
        if (rolling_count ==SAMPLE_RANGE) {
            rolling_count =0;
        }
        if (big_count == 100) {
            big_count=0;
        }
        //ensures that each spin is sampled every iteration
        while (array_counter<(Quantities->array_size*Quantities->array_size)) {
            i = gsl_rng_uniform_int(rng_struct, Quantities->array_size);
            j = gsl_rng_uniform_int(rng_struct, Quantities->array_size);
            if (H(i, j) == 0) {
                Quantities->delta_e = calculate_energy_change(spin, Quantities, i, j);
                if (Quantities->delta_e <= 0.0) {
                    S(i,j) = -S(i, j);
                }
                else{
                    random = gsl_rng_uniform(rng_struct);
                    if (random < exp(-(Quantities->delta_e)/(K_B * Quantities->temperature))) {
                        S(i, j) = -S(i, j);
                    }
                }
                array_counter++;
                H(i, j) = 1;
            }
        }
        
        for (i=0; i<Quantities->array_size; i++) {
            for (j=0; j<Quantities->array_size; j++) {
                H(i, j) = 0;
            }
        }

        if (count%10 == 0) {
            new_energy_val = calculate_energy(spin, Quantities);
            energy_change = new_energy_val - Quantities->total_energy;
            calculate_heat_capacity(Quantities);
            calculate_susceptibility(Quantities);
        }
        count++;
    }
    xfree(has_been_checked);
    return NO_ERROR;
}
//calculates a running average energy over the previous 10 energy values
static Error calculate_average_energy_and_heat_cap (quantities * Quantities, int rolling_count, int count, int big_count, FILE * heatcaptest){
    Quantities->avg_energy[rolling_count] = Quantities->total_energy;
    Quantities->avg_energy_squared[rolling_count] = pow(Quantities->total_energy, 2);
    if (count >=SAMPLE_RANGE) Quantities->avg_heat_capacity[rolling_count] = Quantities->heat_capacity;
    //requires all the arrays to be full before values are calculated and as the heat capacity requires the previous values it needs ten more iterations before the values are meaningful
    if (count%SAMPLE_RANGE == 0 && count != 0) {
        double sum_energy=0;
        double sum_energy_squared=0;
        long double sum_heat_cap=0;
        for (int i=0; i<SAMPLE_RANGE; i++) {
            sum_energy += Quantities->avg_energy[i];
            sum_energy_squared += Quantities->avg_energy_squared[i];
            if (count >=SAMPLE_RANGE) sum_heat_cap += Quantities->avg_heat_capacity[i];
        }
        
        Quantities->current_avg_energy= sum_energy/SAMPLE_RANGE;
        Quantities->current_avg_energy_squared= sum_energy_squared/SAMPLE_RANGE;
        if (count >=SAMPLE_RANGE) Quantities->current_avg_heat_cap= sum_heat_cap/SAMPLE_RANGE;
        if (big_count%10 == 0) {
            Quantities->avg[(big_count/10)] = Quantities->current_avg_heat_cap;
        }
        if (big_count%100 ==0 ) {
            double sum=0;
            for (int x =0; x<SAMPLE_RANGE; x++) {
                sum += Quantities->avg[x];
            }
            Quantities->big_heat_cap = sum/SAMPLE_RANGE;
            
        }
        if (count == (Quantities->number_of_iterations)) {
            fprintf(heatcaptest,"%Lg\t%lg\n", (Quantities->big_heat_cap/pow(Quantities->array_size, 2)), Quantities->temperature);
        }
        
        
    }
    return NO_ERROR;
}

static Error calculate_heat_capacity(quantities * Quantities){
    Quantities->heat_capacity = (1/(pow(Quantities->temperature,2)*K_B)) *(Quantities->current_avg_energy_squared - pow(Quantities->current_avg_energy, 2));
    //printf("Heat_capacity = %.18Lg\n", Quantities->heat_capacity);
    return NO_ERROR;
}

static double calculate_magnetisation(int *spin, quantities * Quantities){
    double magnetisation =0;
    for (int i =0; i<Quantities->array_size; i++) {
        for (int j=0; j<Quantities->array_size; j++) {
            magnetisation += S(i, j);
        }
    }
    magnetisation = magnetisation/(Quantities->array_size*Quantities->array_size);
    return magnetisation;
}

static Error calculate_average_magnetisation_and_susceptibility(quantities * Quantities, int rolling_count, int count){
    Quantities->avg_magnetisation[rolling_count] = Quantities->magnetisation;
    Quantities->avg_magnetistion_squared[rolling_count] = pow(Quantities->magnetisation, 2);
    if (count >=SAMPLE_RANGE) Quantities->avg_susceptibility[rolling_count] = Quantities->susceptibility;
    if (count%10 == 0 && count != 0) {
        double sum_mag=0;
        double sum_mag_squared=0;
        long double sum_susceptibility=0;
        for (int i=0; i<SAMPLE_RANGE; i++) {
            sum_mag += Quantities->avg_magnetisation[i];
            sum_mag_squared +=Quantities->avg_magnetistion_squared[i];
            if (count >=SAMPLE_RANGE) sum_susceptibility += Quantities->avg_susceptibility[i];
        }
        
        Quantities->current_avg_magnetisation= sum_mag/SAMPLE_RANGE;
        Quantities->current_avg_magnetisation_squared= sum_mag_squared/SAMPLE_RANGE;
        if (count >=SAMPLE_RANGE) Quantities->current_avg_susceptibility = sum_susceptibility/SAMPLE_RANGE;

    }
    return NO_ERROR;
}

static Error calculate_susceptibility(quantities * Quantities){
    Quantities->susceptibility = 1/(K_B*Quantities->temperature) * (Quantities->current_avg_magnetisation_squared - pow(Quantities->current_avg_magnetisation, 2));
    //printf("susceptibility is %.18Lg\n", Quantities->susceptibility);
    return NO_ERROR;
}

static Error print_final_vals(quantities * Quantities){
    FILE * final_output =fopen("final_out.txt", "w");
    fprintf(final_output, "Final Energy is : %Lg\n", Quantities->total_energy);
    fprintf(final_output, "Final magnetisation is : %lg\n", Quantities->magnetisation);
    fprintf(final_output, "Final heat capacity is : %Lg\n", Quantities->heat_capacity);
    fprintf(final_output, "Final susceptibility is : %Lg\n", Quantities->susceptibility);
    fprintf(final_output, "Final avg heat cap is : %Lg\n", Quantities->current_avg_heat_cap);
    fprintf(final_output, "Final avg susceptibility is : %Lg\n", Quantities->current_avg_susceptibility);
    fclose(final_output);
    return NO_ERROR;
}

static Error create_and_fill_final_array(quantities * Quantities,int * spin, char * final_file_name){
    FILE * final=NULL;
    static char final_name[MAX_FILE_NAME_LENGTH];
    sprintf(final_name,"T=%lg_%s",Quantities->temperature, final_file_name);
    final = fopen(final_name, "w");
    if (final == NULL) {
        printf("Error opening file\n");
        exit(BAD_FILE_OPEN);
    }
    for (int i=0; i<Quantities->array_size; i++) {
        for (int j=0; j<Quantities->array_size; j++) {
            fprintf(final, "%d\t", S(i, j));
        }
        fprintf(final, "\n");
    }
    fclose(final);
    return NO_ERROR;
}
