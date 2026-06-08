import argparse
import json
import os
import sys
import traceback

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ai.evolutionary import EvolutionaryTrainer
from src.data import db

PROGRESS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "training_progress.json")

def write_progress(status: str, playstyle: str, gen: int, total_gens: int, top_fit: float = 0.0, avg_fit: float = 0.0, best_model: dict = None, error: str = None):
    """Writes the current training state to a JSON file for the UI to read."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    progress_data = {
        "status": status,
        "playstyle": playstyle,
        "current_generation": gen,
        "total_generations": total_gens,
        "top_fitness": top_fit,
        "avg_fitness": avg_fit,
        "best_model": best_model,
        "error": error
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Fantasy Football AI Evolutionary Trainer Daemon")
    parser.add_argument("--playstyle", type=str, default="hybrid", help="Playstyle preset to train (or 'hybrid')")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations to run")
    parser.add_argument("--seasons", type=int, default=10, help="Seasons simulated per evaluation")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size (must be multiples of 10)")
    
    args = parser.parse_args()
    
    playstyle = args.playstyle
    gens = args.generations
    seasons = args.seasons
    pop_size = args.pop_size
    
    print(f"Starting background training run for playstyle: {playstyle}")
    print(f"Params: pop_size={pop_size}, generations={gens}, seasons_per_eval={seasons}")
    
    # Initialize progress file
    write_progress("running", playstyle, 0, gens)
    
    try:
        # Initialize DB
        db.init_db()
        
        # Verify cached years are available
        cached_years = [y for y in [2025, 2024, 2023, 2022] if db.is_year_cached(y, "stats")]
        if not cached_years:
            # Scrape 2025 as fallback if DB is completely empty
            from src.data import scraper
            print("No cached data found! Scraping 2025 season as baseline...")
            scraper.scrape_and_cache_season(2025)
            cached_years = [2025]
            
        print(f"Training on historical years: {cached_years}")
        
        trainer = EvolutionaryTrainer(population_size=pop_size, cached_years=cached_years)
        
        # Define progress callback
        def progress_callback(g, total_g, msg):
            # Print to stdout
            print(f"Gen {g}/{total_g}: {msg}")
            
        # Hook into trainer's loop
        trainer.initialize_population(playstyle)
        
        for g in range(1, gens + 1):
            # Run generation
            import random
            random.shuffle(trainer.population)
            
            # Parallel evaluate
            fitness_scores = trainer._evaluate_fitness_parallel_clean(trainer.population, seasons, playstyle)
            
            # Map
            fitness_map = {idx: score for idx, score in enumerate(fitness_scores)}
            sorted_indices = sorted(list(fitness_map.keys()), key=lambda idx: fitness_map[idx], reverse=True)
            sorted_pop = [trainer.population[idx] for idx in sorted_indices]
            
            top_score = fitness_map[sorted_indices[0]]
            avg_score = sum(fitness_map.values()) / len(fitness_map)
            best_model_dict = sorted_pop[0].__dict__
            
            # Write to progress file
            write_progress("running", playstyle, g, gens, top_score, avg_score, best_model_dict)
            
            # Crossover & Mutate
            survivors = sorted_pop[:pop_size // 2]
            offspring = []
            while len(survivors) + len(offspring) < pop_size:
                parent_a = random.choice(survivors)
                parent_b = random.choice(survivors)
                child = trainer.crossover(parent_a, parent_b, playstyle=playstyle)
                child = trainer.mutate(child, playstyle=playstyle)
                offspring.append(child)
                
            trainer.population = survivors + offspring
            
        # Training complete, save final best model
        best_agent = trainer.population[0]
        model_name = f"Optimized {playstyle.replace('_', ' ').title()}"
        db.save_trained_model(model_name, best_agent.__dict__)
        
        write_progress("completed", playstyle, gens, gens, top_score, avg_score, best_agent.__dict__)
        print("Training successfully complete!")
        
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"Error in training run: {err_msg}")
        write_progress("failed", playstyle, 0, gens, error=str(e) + "\n" + err_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
