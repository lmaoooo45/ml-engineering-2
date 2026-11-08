import argparse

import torch

from movie_assistant.catalog import load_catalog, verify_files
from movie_assistant.recommendations import MovieRecommender, load_ratings, temporal_split
from movie_assistant.settings import storage_root
from movie_assistant.two_tower import TwoTower


def main():
    parser = argparse.ArgumentParser(description='Обучить Two-Tower на ранних оценках и сохранить веса.')
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = storage_root()
    directory = root / "data/ml-100k"
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    verify_files(directory)
    movies = load_catalog(directory)
    train, _ = temporal_split(load_ratings(directory))
    model = TwoTower(MovieRecommender(movies, train), train)
    for epoch, loss in model.fit(args.epochs, args.seed):
        print(f"Эпоха {epoch}, ошибка ранжирования: {loss:.4f}", flush=True)
    path = root / "models/two-tower.pt"
    model.save(path, args.epochs, args.seed)
    print("Веса сохранены:", path)


if __name__ == "__main__":
    main()
