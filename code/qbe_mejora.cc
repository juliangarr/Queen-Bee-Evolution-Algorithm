extern "C" {
#include "cec17.h"
}
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <algorithm>  // para std::min_element
#include <iomanip>    // para std::setprecision, std::fixed y std::scientific

using namespace std;

// Funciones
vector<double> seleccionTorneo(const vector<vector<double>>& population, const vector<double>& fitnesses, int excludeIndex, int numTorneos, std::mt19937 &gen);
void seleccionPadresMejora(int numPadres, const vector<vector<double>>& population, vector<vector<double>>& padres, const vector<double>& fitnesses, int excludeIndex);
void cruce(const vector<vector<double>>& padres, vector<vector<double>>& sucesores, const vector<double>& reina, int numPadres, int dim, double alpha);
void mutacion(vector<vector<double>>& sucesores, double pm, double pm_strong, double xi, int n, int dim, std::mt19937 &gen);
double euclideanDistance(const vector<double>& a, const vector<double>& b);

int main (int argc, char* argv[]) {
    
  // Verificar nº de argumentos
  if (argc < 1 || argc > 2) {
      cerr << "ERROR: Nº de argumentos NO VALIDO. Uso correcto: " << argv[0] << " <dim> <semilla (opcional)>" << endl;
      return 1;
  }

  // Obtener dimensión
  int dim = atoi(argv[1]);
  
  // Pasamos la dimension a char
  std::ostringstream oss;
  oss << dim;
  string dimension_str = oss.str();

  // Nombre del algoritmo a CHAR
  string algname_str = "qbe_Mejora" + dimension_str;
  const char* alg_name = algname_str.c_str();


  // Verificar dimensión
  if (dim != 10 && dim != 30 && dim != 50) {
      cerr << "ERROR: Dimensión NO VALIDA. Debe ser 10, 30 o 50." << endl;
      return 1;
  }

  // Obtener semilla
  int seed = 42;
  if (argc == 3) {
      seed = atoi(argv[2]);
  }

  // Inicializar semilla
  std::mt19937 gen(seed);

  // Inicializar distribución uniforme [-100, 100]
  std::uniform_real_distribution<> uniforme(-100.0, 100.0);  
  
  // PARA CADA FUNCIÓN
  for (int funcid = 1; funcid <= 30; funcid++) {
    // Inicializar función
    cec17_init(alg_name, funcid, dim);
    //cerr <<"Warning: output by console, if you want to create the output file you have to comment cec17_print_output()" <<endl;
    //cec17_print_output(); // Comment to generate the output file

    // Parámetros de ejecución
    int maxevals = 10000*dim; // Número máximo de evaluaciones
    int evals = 0; // Número de evaluaciones
    int numTorneos = 3; // Número de torneos para selección

    // ----------------------------------------- INICIO Q.B.E. Algorithm -----------------------------------------
    
    // Inicialización de parámetros del algoritmo
    int n = 50; // Tamaño de la población de abejas
    int m = n/2; // Número de abejas exploradoras
    double alpha = 0.6; // Parámetro de cruce

    double pm = 0.08; // Probabilidad de mutación normal
    double pm_strong = 0.8; // Probabilidad de mutación fuerte
    double xi = 0.6; // Tasa de mutación normal

    // Declaración de variables
    vector< vector<double> > population(n, vector<double>(dim)); // Población de abejas
    vector<double> fitnesses(n);
    vector<double> best_sol(dim);
    double best_fitness;
    vector<double>::iterator min_it, max_it;
    int min_index, max_index, best_index;
    vector<vector<double>> padres(m); // Crear el vector de padres (zánganos)
    vector<vector<double>> sucesores(n, vector<double>(dim)); // Crear el vector de sucesores

    // ------------------------------ MEJORA ---------------------------------
    vector<double> distances_to_queen(n);

    // Inicialización de la población (aleatoria)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < dim; ++j) {
        population[i][j] = uniforme(gen);
      }
    }
    
    // Evaluación de la población
    for (int i = 0; i < n; ++i) {
      fitnesses[i] = cec17_fitness(&population[i][0]);
      ++evals;
    }
    
    // Selección de la abeja reina (mejor solución)
    min_it = min_element(fitnesses.begin(), fitnesses.end()); // Encontrar el iterador al mínimo valor usando std::min_element
    best_index = distance(fitnesses.begin(), min_it); // Calcular el índice a partir del iterador
    best_fitness = *min_it; // Guardar el mejor fitness
    best_sol = population[best_index]; // Guardar la mejor solución

    // Inicialización de las distancias a la reina
    for (int i = 0; i < n; ++i) {
      distances_to_queen[i] = euclideanDistance(population[i], best_sol);
    }
    
    while (evals < maxevals) {
      // Generación de los padres (zánganos)
      seleccionPadresMejora(m, population, padres, fitnesses, best_index);

      // Cruce de los padres
      cruce(padres, sucesores, best_sol, m, dim, alpha);

      // Mutación para generar los hijos
      mutacion(sucesores, pm, pm_strong, xi, n, dim, gen);

      // Reemplazo de la población CON ELITISMO
        // Reemplazar la población por los sucesores
      population = move(sucesores);
      sucesores.resize(n, vector<double>(dim));

      // Evaluación de la nueva población
      for (int i = 0; i < n; ++i) {
        fitnesses[i] = cec17_fitness(&population[i][0]);
        ++evals;
      }

      // Actualizar el mejor individuo
      min_it = min_element(fitnesses.begin(), fitnesses.end());
      min_index = distance(fitnesses.begin(), min_it);
      if (*min_it < best_fitness) {
        best_fitness = *min_it;
        best_index = min_index;
        best_sol = population[best_index];
      }
      else{
        // Reemplazar el peor individuo por el mejor
          // Encontrar el índice del peor individuo
        max_it = max_element(fitnesses.begin(), fitnesses.end());
        max_index = distance(fitnesses.begin(), max_it);
          // Actualizar el peor individuo por el mejor
        population[max_index] = best_sol;
        fitnesses[max_index] = best_fitness;
        best_index = max_index;
      }
    }

    //cout <<"Best QBE[F" <<funcid <<"]: " << scientific <<cec17_error(best_fitness) <<endl;
    cout << scientific << setprecision(2) << cec17_error(best_fitness) << endl;
  }
}

void seleccionPadresMejora(int numPadres, const vector<vector<double>>& population, vector<vector<double>>& padres, const vector<double>& fitnesses, int excludeIndex ) {
    // Tamaño de la población
    int n = population.size();

    // Seleccionar la reina
    vector<double> reina = population[excludeIndex];
    
    // Seleccionar los m mejores individuos ordenados por su distancia a la reina menos su fitness -> Seleccionar los más alejados con mejor fitness
    vector<pair<double, int>> individuals(n);
    for(int i = 0; i < n; ++i) {
        individuals[i] = make_pair(euclideanDistance(population[i], reina) - fitnesses[i], i);
    }

    // Excluimos a la reina de la selección
    individuals[excludeIndex] = make_pair(__DBL_MIN__, excludeIndex);

    // Ordenamos los individuos por la distancia a la reina menos su fitness en orden descendente
    sort(individuals.begin(), individuals.end(), greater<pair<double, int>>());

    // Seleccionamos los numPadres primeros individuos
    for(int i = 0; i < numPadres; ++i) {
        padres[i] = population[individuals[i].second];
    }
}

void cruce(const vector<vector<double>>& padres, vector<vector<double>>& sucesores, const vector<double>& reina, int numPadres, int dim, double alpha) {
  // CRUCE ARITMÉTICO
    double one_minus_alpha = 1.0 - alpha;
    int index_par, index_impar;
    // Cruzamos todos los zánganos con la reina
    for(int i = 0; i < numPadres; ++i) {
        index_par = 2*i;
        index_impar = 2*i + 1;
        // Calculamos los hijos (cada gen)
        for(int j = 0; j < dim; ++j) {
            sucesores[index_par][j] = alpha * reina[j] + one_minus_alpha * padres[i][j];
            sucesores[index_impar][j] = alpha * padres[i][j] + one_minus_alpha * reina[j];
        }
    }
}


// SIEMPRE MUTAN TODOS LOS INDIVIDUOS, VARÍA EL NÚMERO DE GENES MUTADOS
// (si no mutasen siempre y ALPHA = 0.5 tendríamos soluciones repetidas)
void mutacion(vector<vector<double>>& sucesores, double pm, double pm_strong, double xi, int n, int dim, std::mt19937 &gen) {
    // Distribución normal para generar mutaciones sobre todo en el rango [-50, 50]
    std::normal_distribution<double> normal(0.0, 50.0);

    // Indice de separación de mutación normal y fuerte
    int threshold_normal_mut = floor(xi * n);

    // Número de genes a mutar normal y fuerte
    int num_normal_mut = round(pm * dim);
    int num_strong_mut = round(pm_strong * dim);

    // Vectores de índices de genes para mutación normal y fuerte
    vector<int> index_normal(dim);
    vector<int> index_strong(dim);
    for(int i = 0; i < dim; ++i) {
        index_normal[i] = i;
        index_strong[i] = i;  
    }
    shuffle(index_normal.begin(), index_normal.end(), gen); 
    shuffle(index_strong.begin(), index_strong.end(), gen);

    // Mutación normal para xi*n individuos con prob. pm
    for(int i = 0; i < threshold_normal_mut; ++i) {
        for(int j = 0; j < num_normal_mut; ++j) {
            sucesores[i][index_normal[j]] += normal(gen);
            // Truncar al intervalo [-100, 100]
            sucesores[i][index_normal[j]] = max(-100.0, min(100.0, sucesores[i][index_normal[j]]));
        }
    }

    // Mutación fuerte para el resto de individuos con prob. pm_strong   
    for(int i = threshold_normal_mut; i < n; ++i) {
        for(int j = 0; j < num_strong_mut; ++j) {
            sucesores[i][index_strong[j]] += normal(gen);
            // Truncar al intervalo [-100, 100]
            sucesores[i][index_strong[j]] = max(-100.0, min(100.0, sucesores[i][index_strong[j]]));
        }
    }
}

double euclideanDistance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}