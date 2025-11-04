# ga_feature_selection.py
import numpy as np
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

@dataclass
class GAConfig:
    population_size: int = 80
    n_generations: int = 40
    crossover_rate: float = 0.9
    mutation_rate: float = 0.02
    tournament_k: int = 3
    alpha: float = 1.0   # weight for accuracy
    beta: float = 0.5    # penalty weight for feature ratio
    cv_folds: int = 5
    random_state: int = 42
    min_features: int = 1  # enforce at least 1 feature selected

class GAFeatureSelector:
    def __init__(self, estimator: BaseEstimator, config: GAConfig):
        self.estimator = estimator
        self.config = config
        random.seed(config.random_state)
        np.random.seed(config.random_state)
        self.history = []  # store (generation, best_fitness, mean_fitness, best_mask)

    def _init_population(self, n_features: int) -> np.ndarray:
        # Start with diverse masks; ensure at least one feature
        pop = []
        for _ in range(self.config.population_size):
            mask = np.random.choice([0, 1], size=n_features, p=[0.6, 0.4])
            if mask.sum() < self.config.min_features:
                idx = np.random.randint(0, n_features)
                mask[idx] = 1
            pop.append(mask)
        return np.array(pop, dtype=int)

    def _evaluate_mask(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
        # Ensure at least min_features
        if mask.sum() < self.config.min_features:
            # Heavy penalty
            return 0.0, 0.0, -1.0

        X_sel = X[:, mask.astype(bool)]
        # Build pipeline: scaling for linear models; RF is robust but scaling is harmless
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=False) if hasattr(self.estimator, "sparse_input_") else StandardScaler()),
            ("clf", self.estimator)
        ])

        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        scores = cross_val_score(model, X_sel, y, cv=cv, scoring="accuracy", n_jobs=None)
        acc = float(np.mean(scores))

        feat_ratio = float(mask.sum() / mask.size)
        fitness = self.config.alpha * acc - self.config.beta * feat_ratio
        return acc, feat_ratio, fitness

    def _tournament_select(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        # Select one parent via tournament
        participants = np.random.choice(len(population), self.config.tournament_k, replace=False)
        winner_idx = participants[np.argmax(fitnesses[participants])]
        return population[winner_idx]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        # Uniform crossover
        mask = np.random.rand(parent1.size) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1.astype(int), child2.astype(int)

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        for i in range(individual.size):
            if random.random() < self.config.mutation_rate:
                individual[i] = 1 - individual[i]
        # Ensure at least min_features
        if individual.sum() < self.config.min_features:
            idx = np.random.randint(0, individual.size)
            individual[idx] = 1
        return individual

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        n_features = X.shape[1]
        population = self._init_population(n_features)

        # Evaluate initial population
        accuracies = []
        ratios = []
        fitnesses = []
        for ind in population:
            acc, ratio, fit = self._evaluate_mask(X, y, ind)
            accuracies.append(acc)
            ratios.append(ratio)
            fitnesses.append(fit)

        accuracies = np.array(accuracies)
        ratios = np.array(ratios)
        fitnesses = np.array(fitnesses)

        best_idx = int(np.argmax(fitnesses))
        best_mask = population[best_idx].copy()
        best_fitness = float(fitnesses[best_idx])

        self.history.append({
            "generation": 0,
            "best_fitness": best_fitness,
            "mean_fitness": float(np.mean(fitnesses)),
            "best_mask": best_mask.copy()
        })

        # Evolution loop
        for gen in range(1, self.config.n_generations + 1):
            new_population = []
            # Elitism: keep top 2
            elite_indices = np.argsort(fitnesses)[-2:]
            for ei in elite_indices:
                new_population.append(population[ei].copy())

            # Create rest via selection, crossover, mutation
            while len(new_population) < self.config.population_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.config.population_size:
                    new_population.append(c2)

            population = np.array(new_population, dtype=int)

            # Evaluate
            accuracies = []
            ratios = []
            fitnesses = []
            for ind in population:
                acc, ratio, fit = self._evaluate_mask(X, y, ind)
                accuracies.append(acc)
                ratios.append(ratio)
                fitnesses.append(fit)
            accuracies = np.array(accuracies)
            ratios = np.array(ratios)
            fitnesses = np.array(fitnesses)

            # Track best
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_mask = population[gen_best_idx].copy()
            gen_best_fitness = float(fitnesses[gen_best_idx])

            # Update global best
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_mask = gen_best_mask.copy()

            self.history.append({
                "generation": gen,
                "best_fitness": gen_best_fitness,
                "mean_fitness": float(np.mean(fitnesses)),
                "best_mask": gen_best_mask.copy()
            })

        # Final evaluation of best
        final_acc, final_ratio, _ = self._evaluate_mask(X, y, best_mask)

        return {
            "best_mask": best_mask,
            "selected_indices": np.where(best_mask == 1)[0].tolist(),
            "selected_ratio": float(best_mask.sum() / n_features),
            "cv_accuracy": final_acc,
            "config": self.config,
            "history": self.history
        }

def build_estimator(name: str = "logreg") -> BaseEstimator:
    if name == "logreg":
        return LogisticRegression(max_iter=500, solver="liblinear")
    elif name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError(f"Unknown estimator '{name}'")

# Example usage with a public dataset (Wine)
if __name__ == "__main__":
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt

    data = load_wine()
    X, y = data.data, data.target

    est = build_estimator("logreg")
    cfg = GAConfig(
        population_size=60,
        n_generations=30,
        crossover_rate=0.9,
        mutation_rate=0.03,
        tournament_k=3,
        alpha=1.0,
        beta=0.4,
        cv_folds=5,
        random_state=42,
    )

    ga = GAFeatureSelector(est, cfg)
    result = ga.fit(X, y)

    print("Selected feature indices:", result["selected_indices"])
    print("Selected ratio:", round(result["selected_ratio"], 3))
    print("Cross-validated accuracy:", round(result["cv_accuracy"], 3))

    # Plot fitness over generations
    gens = [h["generation"] for h in result["history"]]
    bests = [h["best_fitness"] for h in result["history"]]
    means = [h["mean_fitness"] for h in result["history"]]
    plt.figure(figsize=(8,4))
    plt.plot(gens, bests, label="Best fitness")
    plt.plot(gens, means, label="Mean fitness", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Fitness evolution")
    plt.legend()
    plt.tight_layout()
    plt.show()
