"""From-scratch Genetic Algorithm Hyperparameter Tuner for RNN/LSTM/GRU.

Implements chromosome encoding, tournament selection, single-point crossover,
gene mutation, and validation RMSE fitness minimization.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any

import numpy as np
import tensorflow as tf

from src.data.windowing import create_sliding_windows
from src.evaluation.metrics import root_mean_squared_error
from src.models.rnn_models import build_rnn_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Search space definition
GENE_SPACE = {
    "window_size": [12, 24, 48],
    "units_l1": [32, 64, 96],
    "units_l2": [16, 32, 64],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [0.0005, 0.001, 0.002],
    "batch_size": [32, 64],
}


class Individual:
    """Represents a candidate hyperparameter chromosome and its fitness."""

    def __init__(self, genes: dict[str, Any]):
        self.genes = genes
        self.fitness: float = float("inf")  # Fitness is validation RMSE (lower is better)

    @classmethod
    def random(cls) -> Individual:
        """Create a random individual from GENE_SPACE."""
        genes = {k: random.choice(v) for k, v in GENE_SPACE.items()}
        return cls(genes)


class GeneticAlgorithmTuner:
    """Genetic Algorithm Optimizer for deep forecasting architectures."""

    def __init__(
        self,
        population_size: int = 8,
        generations: int = 4,
        tournament_size: int = 2,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.25,
        cell_type: str = "LSTM",
        epochs_per_eval: int = 8,
        seed: int = 42,
    ):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.cell_type = cell_type
        self.epochs_per_eval = epochs_per_eval
        self.seed = seed
        self.history: list[dict[str, Any]] = []

        random.seed(seed)
        np.random.seed(seed)

    def evaluate_fitness(
        self,
        ind: Individual,
        train_data: np.ndarray,
        val_data: np.ndarray,
        target_scaler,
    ) -> float:
        """Train model candidate and evaluate inverse-scaled RMSE on validation set."""
        tf.random.set_seed(self.seed)
        w_size = ind.genes["window_size"]

        # Generate windowed arrays
        X_tr, y_tr = create_sliding_windows(train_data, target_idx=0, window_size=w_size, horizon=1)
        X_va, y_va = create_sliding_windows(val_data, target_idx=0, window_size=w_size, horizon=1)

        # Quick evaluation on sample if train data is very large
        sample_limit = 10000
        if len(X_tr) > sample_limit:
            X_tr, y_tr = X_tr[-sample_limit:], y_tr[-sample_limit:]

        model = build_rnn_model(
            cell_type=self.cell_type,
            input_shape=(w_size, train_data.shape[1]),
            layer_units=[ind.genes["units_l1"], ind.genes["units_l2"]],
            dropout_rate=ind.genes["dropout"],
            learning_rate=ind.genes["learning_rate"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]

        model.fit(
            X_tr,
            y_tr,
            validation_data=(X_va, y_va),
            epochs=self.epochs_per_eval,
            batch_size=ind.genes["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )

        preds_scaled = model.predict(X_va, verbose=0)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        actuals = target_scaler.inverse_transform(y_va.reshape(-1, 1)).ravel()

        rmse = root_mean_squared_error(actuals, preds)
        ind.fitness = rmse
        return rmse

    def tournament_selection(self, population: list[Individual]) -> Individual:
        """Select best individual among a random tournament pool."""
        contestants = random.sample(population, self.tournament_size)
        return min(contestants, key=lambda ind: ind.fitness)

    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        """Perform uniform crossover on parent genes."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(p1), copy.deepcopy(p2)

        c1_genes = {}
        c2_genes = {}
        for k in GENE_SPACE:
            if random.random() < 0.5:
                c1_genes[k] = p1.genes[k]
                c2_genes[k] = p2.genes[k]
            else:
                c1_genes[k] = p2.genes[k]
                c2_genes[k] = p1.genes[k]

        return Individual(c1_genes), Individual(c2_genes)

    def mutate(self, ind: Individual) -> Individual:
        """Mutate individual genes with probability mutation_rate."""
        for k, choices in GENE_SPACE.items():
            if random.random() < self.mutation_rate:
                ind.genes[k] = random.choice(choices)
        return ind

    def optimize(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        target_scaler,
    ) -> tuple[Individual, list[dict[str, Any]]]:
        """Execute Genetic Algorithm evolution."""
        logger.info(f"Initializing GA population of size {self.population_size}...")
        population = [Individual.random() for _ in range(self.population_size)]

        best_overall = None

        for gen in range(self.generations):
            logger.info(f"--- GA Generation {gen + 1}/{self.generations} ---")
            for i, ind in enumerate(population):
                if ind.fitness == float("inf"):
                    fitness = self.evaluate_fitness(ind, train_data, val_data, target_scaler)
                    logger.info(f"Gen {gen+1} Ind {i+1}: RMSE={fitness:.2f} MW, Params={ind.genes}")

            # Track generation best
            current_best = min(population, key=lambda ind: ind.fitness)
            if best_overall is None or current_best.fitness < best_overall.fitness:
                best_overall = copy.deepcopy(current_best)

            gen_stats = {
                "generation": gen + 1,
                "best_fitness": current_best.fitness,
                "best_genes": copy.deepcopy(current_best.genes),
                "mean_fitness": np.mean([ind.fitness for ind in population]),
            }
            self.history.append(gen_stats)
            logger.info(f"Generation {gen+1} Summary: Best RMSE={current_best.fitness:.2f} MW")

            # Create next generation
            next_pop = [copy.deepcopy(best_overall)]  # Elitism: carry best forward
            while len(next_pop) < self.population_size:
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)
                c1, c2 = self.crossover(p1, p2)
                next_pop.append(self.mutate(c1))
                if len(next_pop) < self.population_size:
                    next_pop.append(self.mutate(c2))

            population = next_pop

        return best_overall, self.history
