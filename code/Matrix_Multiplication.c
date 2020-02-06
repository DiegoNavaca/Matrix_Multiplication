#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// The basic function for multiply matrices
void basic_multiplication(double * matrix_1, unsigned int r_1, unsigned int c_1, double * matrix_2, unsigned int r_2, unsigned int c_2, double * result){
  if(c_1 == r_2){     // Check if these matrices can be multiplied
    // For every position in the matrix result
    for(int r = 0; r < r_1; ++r){
      for(int c = 0; c < c_2; ++c){
	// Add the product of the corresponding positions of the other matrices
	result[c + r*c_2] = 0;
	for(int k = 0; k < r_2; ++k){
	  result[c + r*c_2] += matrix_1[k + r*c_1]*matrix_2[c + k*c_2];	  
	}	
      }
    }
  }
  else{
    fprintf(stderr,"Error: Wrong matrix size\n");
    exit(-1);
  }
}

// The same basic algorithm but with some tweaks for a more optimiced performance
void optimiced_multiplication(double * matrix_1, unsigned int r_1, unsigned int c_1, double * matrix_2, unsigned int r_2, unsigned int c_2, double * result){
  double * aux = (double*)malloc(r_2*c_2*sizeof(double));
  if(c_1 == r_2){
    // We copy matrix_2 into an auxiliar matrix to better take advantage of the cache during execution
   for(int r = 0; r < r_2; ++r){
      for(int c = 0; c < c_2; ++c){
	// In the new matrix, each row is a column of matrix_2
        aux[r + c*r_2] = matrix_2[c + r*c_2];
      }
   }
   // We can do a tiny loop unwinding for a minor performance boost
   // We first have to check if r_2 is even or odd
   if(r_2 % 2 == 0){
    for(int r = 0; r < r_1; ++r){
      for(int c = 0; c < c_2; ++c){
	result[c + r*c_2] = 0;
	for(int k = 0; k < r_2; k += 2){
	  result[c + r*c_2] += matrix_1[k + r*c_1]*aux[k + c*c_2];
	  result[c + r*c_2] += matrix_1[k+1 + r*c_1]*aux[k+1 + c*c_2];
	}
      }
    } 
   }else{
    for(int r = 0; r < r_1; ++r){
      for(int c = 0; c < c_2; ++c){
	// If r_2 is odd we have to add the first iteration of the loop 
	result[c + r*c_2] = matrix_1[r*c_1]*aux[c*c_2];
	for(int k = 1; k < r_2; k += 2){
	  result[c + r*c_2] += matrix_1[k + r*c_1]*aux[k + c*c_2];
	  result[c + r*c_2] += matrix_1[k+1 + r*c_1]*aux[k+1 + c*c_2];
	}
      }
    } 
   }
   free(aux);
  }
  else{
    fprintf(stderr,"Error: Wrong matrix size\n");
    exit(-1);
  }
}

// The basic function but prepared for parallel calculation
void parallel_multiplication(double * matrix_1, unsigned int r_1, unsigned int c_1, double * matrix_2, unsigned int r_2, unsigned int c_2, double * result){
  if(c_1 == r_2){
#pragma omp parallel for if(r_1*c_1 > 100)    // This will divide the result matrix into rows and divide them into the threads 
    for(int r = 0; r < r_1; ++r){
      for(int c = 0; c < c_2; ++c){
	result[c + r*c_2] = 0;
	for(int k = 0; k < r_2; ++k){
	  result[c + r*c_2] += matrix_1[k + r*c_1]*matrix_2[c + k*c_2];	  
	}	
      }
    }
  }
  else{
    fprintf(stderr,"Error: Wrong matrix size\n");
    exit(-1);
  }
}

// The combination of the last two functions
void optimiced_parallel_multiplication(double * matrix_1, unsigned int r_1, unsigned int c_1, double * matrix_2, unsigned int r_2, unsigned int c_2, double * result){
  double * aux = (double*)malloc(r_2*c_2*sizeof(double));
  if(c_1 == r_2){
#pragma omp parallel if(r_1*c_1 > 100)
    {
      #pragma omp for
      for(int r = 0; r < r_2; ++r){
	for(int c = 0; c < c_2; ++c){
	  aux[r + c*r_2] = matrix_2[c + r*c_2];
	}
      }
      if(r_2 % 2 == 0){
	#pragma omp for
	for(int r = 0; r < r_1; ++r){
	  for(int c = 0; c < c_2; ++c){
	    result[c + r*c_2] = 0;
	    for(int k = 1; k < r_2; k += 2){
	      result[c + r*c_2] += matrix_1[k + r*c_1]*aux[k + c*c_2];
	      result[c + r*c_2] += matrix_1[k-1 + r*c_1]*aux[k-1 + c*c_2];
	    }
	  }
	} 
      }else{
	#pragma omp for
	for(int r = 0; r < r_1; ++r){
	  for(int c = 0; c < c_2; ++c){
	    result[c + r*c_2] = 0;
	    for(int k = 1; k < r_2; k += 2){
	      result[c + r*c_2] += matrix_1[k + r*c_1]*aux[k + c*c_2];
	      result[c + r*c_2] += matrix_1[k-1 + r*c_1]*aux[k-1 + c*c_2];
	    }
	    result[c + r*c_2] += matrix_1[r_2-1 + r*c_1]*aux[r_2-1 + c*c_2];
	  }
	} 
      }
    }
    free(aux);
  }
  else{
    fprintf(stderr,"Error: Wrong matrix size\n");
    exit(-1);
  }
}

// A function to try the execution and check the times of all multiplication functions
void try_all_functions(double * matrix_1, unsigned int r_1, unsigned int c_1, double * matrix_2, unsigned int r_2, unsigned int c_2, double * result){
  // Variables to estimate the execution time
  double cgt1, cgt2, ncgt;
  
  cgt1 = omp_get_wtime();
  basic_multiplication(matrix_1,r_1,c_1,matrix_2,r_2,c_2, result);
  cgt2 = omp_get_wtime();

  // We print the first values of the matrix to check that everything works as intended
  for(int i = 0; i<c_2 && i<5; ++i){
    printf("%.1f ",result[i]);
  }
  printf("\n");
  // To get the execution time we substract the initial time to the complition time 
  ncgt=(double) (cgt2-cgt1)+
       (double) ((cgt2-cgt1)/(1.e+9));
  printf("Normal Time: %11.9f\n\n",ncgt);

  // We reapeat for every function
  cgt1 = omp_get_wtime();
  optimiced_multiplication(matrix_1,r_1,c_1,matrix_2,r_2,c_2, result);
  cgt2 = omp_get_wtime();

  for(int i = 0; i<c_2 && i<5; ++i){
    printf("%.1f ",result[i]);
  }
  printf("\n");
  ncgt=(double) (cgt2-cgt1)+
       (double) ((cgt2-cgt1)/(1.e+9));
  printf("Optimiced Time: %11.9f\n\n",ncgt);

  cgt1 = omp_get_wtime();
  parallel_multiplication(matrix_1,r_1,c_1,matrix_2,r_2,c_2, result);
  cgt2 = omp_get_wtime();

  for(int i = 0; i<c_2 && i<5; ++i){
    printf("%.1f ",result[i]);
  }
  printf("\n");
  ncgt=(double) (cgt2-cgt1)+
       (double) ((cgt2-cgt1)/(1.e+9));
  printf("Parallel Time: %11.9f\n\n",ncgt);

  cgt1 = omp_get_wtime();
  optimiced_parallel_multiplication(matrix_1,r_1,c_1,matrix_2,r_2,c_2, result);
  cgt2 = omp_get_wtime();

  for(int i = 0; i<c_2 && i<5; ++i){
    printf("%.1f ",result[i]);
  }
  printf("\n");
  ncgt=(double) (cgt2-cgt1)+
       (double) ((cgt2-cgt1)/(1.e+9));
  printf("Final Time: %11.9f\n\n\n",ncgt);

}

// A function to just get the time of one function, handy to make charts and graphs
// Function number: 0 = basic, 1 = optimiced, 2 = parallel, 3 = final
void try_single_function(int function_number, double * matrix_1, unsigned int r_1, unsigned int c_1, double * matrix_2, unsigned int r_2, unsigned int c_2, double * result){

  double cgt1, cgt2, ncgt;  
  ncgt = 0;
  
  // To get a more acurate time we execute the function several times and calculate the average
  int rep = 3;
  for(int i = 0; i<rep; ++i){
    switch(function_number){
    case 0:
      cgt1 = omp_get_wtime();
      basic_multiplication(matrix_1,r_1,c_1,matrix_2,r_2,c_2, result);
      cgt2 = omp_get_wtime();
      break;
      case 1:
      cgt1 = omp_get_wtime();
      optimiced_multiplication(matrix_1,r_1,c_1,matrix_2,r_2,c_2, result);
      cgt2 = omp_get_wtime();
      break;
      case 2:
      cgt1 = omp_get_wtime();
      parallel_multiplication(matrix_1,r_1,c_1,matrix_2,r_2,c_2, result);
      cgt2 = omp_get_wtime();
      break;
      case 3:
      cgt1 = omp_get_wtime();
      optimiced_parallel_multiplication(matrix_1,r_1,c_1,matrix_2,r_2,c_2, result);
      cgt2 = omp_get_wtime();
      break;
    }

  ncgt+=(double) (cgt2-cgt1)+
       (double) ((cgt2-cgt1)/(1.e+9));
  }
  ncgt /= rep;
  printf("%11.9f\n",ncgt);
}

int main(int argc, char ** argv){
  // To test this out we are going to use matrices of default size 500x500
  int size;
  if(argc == 2){
    size = atoi(argv[1]);
  }else{
    size = 500;
  }
  const int r_1 = size, r_2 = size, c_1 = size, c_2 = size;
  // We create the matrices using a pointer
  double * matrix_1 = (double*)malloc(r_1*c_1*sizeof(double));
  double * matrix_2 = (double*)malloc(r_2*c_2*sizeof(double));
  double * result = (double*)malloc(r_1*c_2*sizeof(double));

  // We fill the matrices with someting
  for(int r = 0; r < r_1; ++r){
      for(int c = 0; c < c_1; ++c){
	matrix_1[c + r*c_1] = c + r*c_1;
      }
  }
  for(int r = 0; r < r_2; ++r){
      for(int c = 0; c < c_2; ++c){
	matrix_2[c + r*c_2] = c + r*c_2;	
      }
  }
  
   try_all_functions(matrix_1,r_1,c_1,matrix_2,r_2,c_2, result);
  // try_single_function(0,matrix_1,r_1,c_1,matrix_2,r_2,c_2, result);

  // We free the pointer of each matrix
  free(matrix_1);
  free(matrix_2);
  free(result);
  
  return 0;
}

