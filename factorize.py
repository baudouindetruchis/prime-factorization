import sys, os, time, math
import numpy as np
from tqdm import tqdm
from itertools import compress
import pickle
import gc


# ========== FUNCTIONS ==========

def primesfrom2to(n):
    """Faster sieve"""
    sieve = np.ones(n//3 + (n%6==2), dtype=np.bool)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)]

def sieve_eratosthenes_array(n):
    """Return a list of all prime numbers || method: Sieve of Eratosthenes with numpy"""
    # Assume all True at first
    primes = np.ones(n + 1, dtype=bool)

    # Zero and one : not primes
    primes[0] = False
    primes[1] = False

    p = 2
    while (p * p <= n): 
        # If prime[p] is not changed, then it is a prime 
        if (primes[p] == True):
            # Update all multiples of p 
            primes[p*p::p] = False
        p += 1

    # Return indices of True values
    return np.argwhere(primes == True).flatten()

def sieve_eratosthenes_list(n):
    """Return a list of all prime numbers || method: Sieve of Eratosthenes"""
    # Assume all True at first
    primes = [True for i in range(n + 1)] 
    p = 2

    while (p * p <= n): 
        # If prime[p] is not changed, then it is a prime 
        if (primes[p] == True): 
            # Update all multiples of p 
            for i in range(p * 2, n + 1, p): 
                primes[i] = False
        p += 1

    primes[0] = False
    primes[1] = False

    # Return indices of True values
    return list(compress(range(len(primes)), primes))

def primes_loop_odd(n):
    """Create a lookup table for later || method: loop over odd numbers"""
    # Only one even number is a prime, let's add it right away
    primes = [2]
    # Try only odd numbers then
    for possible_prime in tqdm(range(3, n, 2)):
        # Assume number is prime until shown it is not
        is_prime = True
        # Every composite number has a factor less than or equal to its square root
        for i in range(2, int(possible_prime**0.5) + 1):
            # Can be divided by something else than (1 and itself) --> not a prime
            if possible_prime % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(possible_prime)

    return primes

def check_factor(remainder, factors, i):
    """Check if a value is a factor"""
    while remainder % i == 0:
        factors.append(i)
        remainder = remainder//i

    return remainder, factors


# ========== LOOKUP TABLE ==========

total_start_time = time.time()

PRIMES = primesfrom2to(10**8)

print("--- [GEN] %s seconds ---" % (time.time() - total_start_time))


# ========== MAIN ==========

def factorize(numbers):
    """Decompose a number in its prime factors || method: loop over prime numbers using lookup + vectorized testing"""
    
    start_time = time.time()
    results = {}

    for n in numbers:
        print(n, end=' | ')

        # Initialization
        remainder = n
        factors = []

        # Factor less or equal to sqrt(n)
        max_factor = int(np.sqrt(n)) + 1

        # Vectorization
        primes_selected = PRIMES[PRIMES < max_factor]
        factors_index = remainder % primes_selected
        factors_array = primes_selected[np.argwhere(factors_index == 0).flatten()]
        
        for i in list(factors_array):
            # Check a factor
            remainder, factors = check_factor(remainder, factors, i)
            if remainder == 1:
                break

        # Add result to dict
        if remainder != 1:
            factors.append(remainder)
        # Convert from numpy.int64 to int
        results[n] = [int(i) for i in factors]
        print(results[n])
    
    print("--- %s seconds ---" % (time.time() - start_time))

    return results

def factorize_loop(numbers):
    """Decompose a number in its prime factors || method: loop over prime numbers using lookup || benchmark 10**6 max factor: 4s """
    
    start_time = time.time()
    results = {}

    for n in numbers:
        print(n, end=' | ')

        # Initialization
        remainder = n
        factors = []

        racine = int(np.sqrt(remainder))

        # Brute force (test only prime numbers) (np.iter faster iterator : read-only)
        # for i in np.nditer(PRIMES[PRIMES < max_factor]):
        for i in PRIMES:
            # Compare trial with current max factor
            if i > racine:
                break
            # Check a factor
            while remainder % i == 0:
                remainder = remainder // i
                factors.append(i)
                racine = int(np.sqrt(remainder))
        # Add result to dict
        if remainder != 1:
            factors.append(remainder)
        # Convert from numpy.int64 to int
        results[n] = factors
        print(results[n])
    
    print("--- %s seconds ---" % (time.time() - start_time))

    return results

def factorize_odd(numbers):
    """Decompose a number in its prime factors || method: loop over odd numbers || benchmark 10**6 max factor: 21s"""
    
    start_time = time.time()
    results = {}

    for n in numbers:
        print(n, end=' | ')

        # Initialization
        remainder = n
        factors = []

        # Factor less or equal to sqrt(n)
        max_factor = int(n**0.5) + 1
        # Remove factor 2
        remainder, factors = check_factor(remainder, factors, 2)
        # Brute force (test only odd numbers)
        for i in range(3, min(max_factor, 10**6), 2):
            remainder, factors = check_factor(remainder, factors, i)
            # Check if finished (if so save all prime numbers)
            if remainder == 1:
                break
        
        # Add result to dict
        if remainder != 1:
            factors.append(remainder)
        results[n] = factors
        print(results[n])
    
    print("--- %s seconds ---" % (time.time() - start_time))

    return results


#########################################
#### Ne pas modifier le code suivant ####
#########################################
if __name__=="__main__":
    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    
    # un repertoire des fichiers en entree doit être passé en parametre 1
    if not os.path.isdir(input_dir):
	    print(input_dir, "doesn't exist")
	    exit()

    # un repertoire pour enregistrer les résultats doit être passé en parametre 2
    if not os.path.isdir(output_dir):
	    print(input_dir, "doesn't exist")
	    exit()       

     # Pour chacun des fichiers en entrée 
    for data_filename in sorted(os.listdir(input_dir)):
        # importer la liste des nombres
        data_file = open(os.path.join(input_dir, data_filename), "r")
        numbers = [int(line) for line in data_file.readlines()]        
        
        # decomposition en facteurs premiers
        D = factorize(numbers)

        # fichier des reponses depose dans le output_dir
        output_filename = 'answer_{}'.format(data_filename)             
        output_file = open(os.path.join(output_dir, output_filename), 'w')
        
        # ecriture des resultats
        for (n, primes) in D.items():
            output_file.write('{} {}\n'.format(n, primes))
        
        output_file.close()