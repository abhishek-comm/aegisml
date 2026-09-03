from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AEGIS_")

    database_url: str = "sqlite:///./data/aegis.db"
    drift_warning_threshold: float = 0.12
    drift_critical_threshold: float = 1.2
    random_seed: int = 42
    model_dir: Path = Path("./data/models")


settings = Settings()
