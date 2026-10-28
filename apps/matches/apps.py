from django.apps import AppConfig


class MatchesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.matches"
    verbose_name = "Matches"

    def ready(self):
        """Load the ML model once at server startup (§7.4)."""
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
            from ml.predict import warm_up
            warm_up()
        except FileNotFoundError as e:
            logging.getLogger(__name__).warning("Model not trained yet: %s", e)
        except Exception as e:
            logging.getLogger(__name__).warning("Model warmup skipped: %s", e)