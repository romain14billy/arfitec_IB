"""Chargement des configurations d'expérience depuis le dossier experiments."""

import importlib


def load_config(experiment):
    """Charge la configuration d'une expérience à partir d'un module Python."""
    config_module = importlib.import_module(f"experiments.{experiment}.config")
    importlib.reload(config_module)
    return config_module.CONFIG

