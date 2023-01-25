"""
WSGI config for human_feedback_site project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.6/howto/deployment/wsgi/
"""

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "human_feedback_site.settings")

from django.core.wsgi import get_wsgi_application
# from whitenoise.django import DjangoWhiteNoise
from whitenoise import WhiteNoise

application = get_wsgi_application()
# application = DjangoWhiteNoise(application)
application = WhiteNoise(application)
