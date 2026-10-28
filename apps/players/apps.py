from django.apps import AppConfig


class PlayersConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.players"
    verbose_name = "Players"

    def ready(self):
        """Load player model in background at startup (lazy-safe)."""
        import logging
        import sys

        skip_commands = {
            "makemigrations", "migrate", "collectstatic", "test",
            "shell", "createsuperuser", "changepassword", "flush",
            "loaddata", "dumpdata",
        }
        if any(cmd in sys.argv for cmd in skip_commands):
            return
        try:
            from ml.predict_player import warm_up
            warm_up()
        except FileNotFoundError as e:
            logging.getLogger(__name__).warning("Player model not trained yet: %s", e)
        except Exception as e:
            logging.getLogger(__name__).warning("Player warmup skipped: %s", e)