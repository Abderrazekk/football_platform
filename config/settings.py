"""
Django settings for the Football Intelligence Platform.

Cahier des charges: Football Intelligence Platform (Tekup, 2026-2027).
Student: Abderazek BEL HADJ MBAREK.

Note: SECRET_KEY is hardcoded for local development only.
It will be moved to .env in Phase 9 (security hardening).
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


# SECURITY WARNING: replace with the value generated above.
SECRET_KEY = "c0oz%u@1=%9m&#as((g7_hh+hs9!p-dzy^f*f_0zz*wd4wk3x8"

# SECURITY WARNING: never True in production.
DEBUG = True

ALLOWED_HOSTS = ["localhost", "127.0.0.1"]


# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
INSTALLED_APPS = [
    # Django built-ins
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party
    "rest_framework",
    # Local apps
    "apps.core",
    "apps.accounts",
    "apps.matches",
    "apps.players",
    "apps.season",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"


# ---------------------------------------------------------------------------
# Database — SQLite per cahier des charges §7.1
# ---------------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# ---------------------------------------------------------------------------
# Internationalization — cahier des charges §11 (interface en français)
# ---------------------------------------------------------------------------
LANGUAGE_CODE = "fr"
TIME_ZONE = "Africa/Tunis"
USE_I18N = True
USE_TZ = True


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]


# ---------------------------------------------------------------------------
# Default primary key
# ---------------------------------------------------------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# ---------------------------------------------------------------------------
# Auth redirects — used from Phase 5 onwards
# ---------------------------------------------------------------------------
LOGIN_URL = "/accounts/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/"