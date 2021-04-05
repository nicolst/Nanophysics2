#include <iostream>
#include <armadillo>

//using namespace std;
using namespace arma;

#define N 30
#define XN 600
#define XMAX 60000

// Imaginary unit
constexpr static cx_double j(0.0, 1.0);

// Common matrices
cx_mat s;
cx_mat incoherentS;
const cx_mat I(N, N, fill::eye);

double alpha = 0.0;
int n = 0; // Amount of coherent conductance experiments to run
double k[N];

// Combines A and B, setting matrix B to this value
void combine(const cx_mat& A, cx_mat& B) {
    cx_mat S(2*N, 2*N, fill::zeros);
    //cx_mat I(N, N, fill::eye);

    int mid = N - 1;
    int end = 2*N - 1;

    // Need a copy of B-matrix in order to not ruin calculations..
    cx_mat Bcopy = B;

    // Use subviews for (hopefully) better performance (avoid copying)
    subview<cx_double> t1 = A(span(0, mid), span(0, mid));
    subview<cx_double> t2 = Bcopy(span(0, mid), span(0, mid));

    subview<cx_double> t1p = A(span(N, end), span(N, end));
    subview<cx_double> t2p = Bcopy(span(N, end), span(N, end));

    subview<cx_double> r1 = A(span(N, end), span(0, mid));
    subview<cx_double> r2 = Bcopy(span(N, end), span(0, mid));

    subview<cx_double> r1p = A(span(0, mid), span(N, end));
    subview<cx_double> r2p = Bcopy(span(0, mid), span(N, end));

    cx_mat inv1 = inv(I- r1p * r2);
    cx_mat inv2 = inv(I - r2 * r1p);

    // Calculate new S_tot and set B to it
    B(span(0, mid), span(0, mid)) = t2 * inv1 * t1;
    B(span(N, end), span(N, end)) = t1p * inv2 * t2p;
    B(span(N, end), span(0, mid)) = r1 + t1p * inv2 * r2 * t1;
    B(span(0, mid), span(N, end)) = r2p + t2 * inv1 * r1p * t2p;
}

// Calculate p-matrix for a given length (between impurities)
cx_mat p(double l) {
    // pm: full p-matrix, psub: one sub-matrix (top left / bottom right)
    cx_mat pm(2*N, 2*N, fill::zeros);

    // Calculate sub-matrix
    cx_mat psub(N, N, fill::zeros);
    for (int i = 0; i < N; ++i) {
        psub(i, i) = exp(j * k[i] * l);
    }

    // Set top left and bottom right equal to same sub-matrix
    pm(span(0, N-1), span(0, N-1)) = psub;
    pm(span(N, 2*N-1), span(N, 2*N-1)) = psub;
    return pm;
}

// Calculate random impurities and return a vector containing lengths between them
vec randomImpurityLengths() {
    vec impurities(XN, fill::randu);
    vec unique_imp = unique(impurities); // Returns ordered vector of unique elements
    // Ensure that we have enough impurities
    while (unique_imp.n_elem != XN) {
        impurities = vec(XN, fill::randu);
        unique_imp = unique(impurities);
    }
    unique_imp = 60000.0 * unique_imp; // Scaling

    // Calculate lengths from impurities
    vec lengths(XN+1, fill::zeros);
    lengths(span(1, XN-1)) = unique_imp(span(1, XN-1)) - unique_imp(span(0, XN-2)); // x_n - x_{n-1} in essence
    lengths(0) = impurities(0);
    lengths(XN) = 60000.0 - impurities(XN-1);
    return lengths;
}

// Calculate conductance for random impurities, coherent case
double calculateConductance() {
    // Get random sample of impurities
    vec lengths = randomImpurityLengths();

    // Calculate S_tot
    cx_mat stot = p(lengths(XN)); 
    for (int i = XN-1; i >= 0; --i) {
        combine(s, stot);
        combine(p(lengths[i]), stot);
    }
    
    // Extract t, and return sum(t* % t), %: element-wise multiplication
    cx_mat t = stot(span(0, N-1), span(0, N-1));
    return accu(conj(t) % t).real();
}

// Calculate conductance for incoherent case, impurities irrelevant
double calculateIncoherentConductance() {
    cx_mat stot = incoherentS;
    for (int i = 1; i < XN; ++i) {
        combine(incoherentS, stot);
    }
    // Extract T and return sum(T)
    cx_mat T = stot(span(0, N-1), span(0, N-1));
    return accu(T).real();
}

// Calculate coherent conductance in parallel (OpenMP)
void parallelCoherent(int n) {
    vec conductances(n, fill::zeros);
#pragma omp parallel num_threads(n)
    {
        arma_rng::set_seed_random();
        int n = omp_get_thread_num();
        conductances(n) = calculateConductance();
        std::cout << "Thread " << n << " finished\n";
    }

    // Print conductances, mean, variance and standard deviation
    std::cout << "Conductances:\n" << conductances;
    std::cout << "Mean: " << mean(conductances) << std::endl;
    std::cout << "Variance: " << var(conductances) << std::endl;
    std::cout << "Standard deviation: " << stddev(conductances) << std::endl;

    // Save conductance values to file, plot data using NumPy/Matplotlib
    conductances.save("coherent.dat", raw_ascii);
}

int main(int argc, char** argv) {
    // Set random seed for uniform distribution
    arma_rng::set_seed_random();

    // Calculate k-values
    for (int i = 0; i < N; ++i) {
        k[i] = sqrt(30.5*30.5 - 1.0*(i+1)*(i+1));
    }

    // Read alpha from argument list
    if (argc > 1) {
        alpha = atof(argv[1]);
    }

    // Read n (amount of experiments) from argument list
    if (argc > 2) {
        n = atoi(argv[2]);
    }

    // Calculate s-matrix, since it is common to all calculations.
    const cx_mat D(2*N, 2*N, fill::ones);
    s = expmat(j * alpha * D);
    incoherentS = conj(s) % s;

    // Calculate coherent conductances if n>0, else calculate incoherent
    if (n > 0) {
        std::cout << "Running " << n << " coherent experiments for alpha=" << alpha << std::endl;
        parallelCoherent(n);
    } else {
        std::cout << "Calculating incoherent conductance for alpha=" << alpha << std::endl;
        std::cout << "Conductance: " << calculateIncoherentConductance() << std::endl; 
    }

    return 0;
}