import django
import logging
from django.conf import settings
from django.core.exceptions import AppRegistryNotReady
from human_feedback_site import settings as site_settings

def initialize():
    try:
        conf = {k:v for k,v in site_settings.__dict__.items() if k.isupper()}
        settings.configure(**conf)
        django.setup()
    except RuntimeError:
        logging.warning("Tried to double configure the API, ignore this if running the Django app directly")

initialize()

try:
    from human_feedback_api.models import Comparison
except AppRegistryNotReady:
    logging.info("Could not yet import Feedback")
