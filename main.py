import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size=100, mutation_rate=0.1):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.create_initial_population()
    
    def create_initial_population(self):
        # 8 genden oluşan bir populasyon oluşturuyoruz
        return [np.random.permutation(8) for _ in range(self.population_size)]

    def fitness(self, chromosome):
        
        conflicts = 0
        
        for i in range(8):
            for j in range(i + 1, 8):
                # Çapraz çatışma var mı kontrol ediyoruz
                if abs(i- j) == abs(chromosome[i] - chromosome[j]):
                    conflicts += 1
        return 1 / (conflicts + 1) # Çatışma sayısı az ise fitness değeri yüksek olacak
    
    def select_parent(self, fitness):
        # Fitness değeri yüksek olan bireyi seçeceğiz
        total_fitness = sum(fitness)
        r = random.uniform(0, total_fitness) # Rastgele bir sayı seçiyoruz
        current_sum = 0
        
        for i, fitness in enumerate(fitness):
            current_sum += fitness
            if current_sum > r:
                return self.population[i]
        return self.population[-1]
    
    def crossover(self, parent1, parent2):
        # Sıralı çaprazlama
        point1, point2 = sorted(random.sample(range(8), 2))
        
        child = [-1] * 8
        # Parent1'den seçilen aralığı child'a kopyala
        child[point1:point2] = parent1[point1:point2]
        
        # Kalan pozisyonları parent2'den doldur
        remaining = [x for x in parent2 if x not in child[point1:point2]]
        j = 0
        for i in range(8):
            if child[i] == -1:
                child[i] = remaining[j]
                j += 1
                
        return np.array(child)
    
    def mutate(self, chromosome):
        
        if random.random() < self.mutation_rate:
            # Rastgele iki pozisyon seçip yer değiştir
            i, j = random.sample(range(8), 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome
    
    def solve(self, max_generations=1000):
        # Belirli bir nesil sayısına (1000) kadar çözüm bulmaya çalışacağız
        for generation in range(max_generations):
            # Fitness değerlerini hesapla
            fitnesses = [self.fitness(chrom) for chrom in self.population]
            best_fitness = max(fitnesses)
            
            # Eğer çözüm bulunduysa bitir
            if best_fitness == 1.0:
                best_index = fitnesses.index(best_fitness)
                return self.population[best_index], generation
            
            # Yeni nesil oluştur
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.select_parent(fitnesses)
                parent2 = self.select_parent(fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            
            if generation % 100 == 0:
                print(f"Nesil {generation}: En iyi fitness = {best_fitness:.4f}")
        
        # Maksimum nesil sayısına ulaşıldı
        best_index = fitnesses.index(max(fitnesses))
        return self.population[best_index], max_generations        
        for generation in range(max_generations):
            fitness_values = [self.fitness(chromosome) for chromosome in self.population]
            new_population = []
            
            for _ in range(self.population_size):
                parent1 = self.select_parent(fitness_values)
                parent2 = self.select_parent(fitness_values)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            if max(fitness_values) == 1:
                break
        
        best_chromosome = self.population[np.argmax(fitness_values)]
        return best_chromosome
        
def print_board(solution):
    # Tahtayı oluşturuyoruz
    board = np.zeros((8, 8), dtype=int)
    for i, pos in enumerate(solution):
        board[pos, i] = 1
    print("\n8 Vezir Çözümü:")
    print("-" * 19)
    for row in board:
        print("|", end=" ")
        for cell in row:
            print("V " if cell else ". ", end="")
        print("|")
    print("-" * 19)

# Algoritmayı çalıştır
ga = GeneticAlgorithm(population_size=100, mutation_rate=0.1)
solution, generation = ga.solve()
print(f"\nÇözüm {generation}. nesilde bulundu")
print_board(solution)