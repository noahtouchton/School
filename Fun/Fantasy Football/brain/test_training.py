import sys
import os

# Add src to the path
sys.path.append("/Users/noahtouchton/School_Git/School/Fun/Fantasy Football")

from src.ai.evolutionary import EvolutionaryTrainer
from src.data import db

def main():
    print("Testing Evolutionary Training Loop...")
    # Initialize DB (make sure 2024 is scraped)
    db.init_db()
    
    # Run a quick 2-generation training run with population size 8 (1 league)
    # to verify that calculations, crossovers, mutations, and simulations execute
    trainer = EvolutionaryTrainer(year=2024, population_size=8)
    
    print("Running 2 Generations of training on 2024 season stats...")
    best_params = trainer.train(generations=2)
    
    print("\nTraining complete!")
    print("Evolved 'Super Expert' Parameters:")
    for field, value in best_params.__dict__.items():
        print(f"  {field}: {value}")

if __name__ == "__main__":
    main()
