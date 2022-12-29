from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import gettext_lazy as _

RESPONSE_KIND_TO_RESPONSES_OPTIONS = {'left_or_right': ['left', 'right', 'tie', 'abstain']}

def validate_inclusion_of_response_kind(value):
    kinds = RESPONSE_KIND_TO_RESPONSES_OPTIONS.keys()
    if value not in kinds:
        raise ValidationError(_('%(value)s is not included in %(kinds)s'), params={'value': value, 'kinds': kinds}, )

class Comparison(models.Model):
    created_at = models.DateTimeField('date created', auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    media_url_1 = models.TextField('media url #1', db_index=True)
    media_url_2 = models.TextField('media url #2', db_index=True)

    shown_to_tasker_at = models.DateTimeField('time shown to tasker', db_index=True, blank=True, null=True)
    responded_at = models.DateTimeField('time response received', db_index=True, blank=True, null=True)
    response_kind = models.TextField('the response from the tasker', db_index=True,
        validators=[validate_inclusion_of_response_kind])
    response = models.TextField('the response from the tasker', db_index=True, blank=True, null=True)
    experiment_name = models.TextField('name of experiment')

    local = models.BooleanField(default=False)

    priority = models.FloatField('site will display higher priority items first', db_index=True)
    note = models.TextField('note to be displayed along with the query', default="", blank=True)

    # Validation
    def full_clean(self, exclude=None, validate_unique=True):
        super(Comparison, self).full_clean(exclude=exclude, validate_unique=validate_unique)
        self.validate_inclusion_of_response()

    @property
    def response_options(self):
        try:
            return RESPONSE_KIND_TO_RESPONSES_OPTIONS[self.response_kind]
        except KeyError:
            raise KeyError("{} is not a valid response_kind. Valid response_kinds are {}".format(self.response_kind,
                RESPONSE_KIND_TO_RESPONSES_OPTIONS.keys()))

    def validate_inclusion_of_response(self):
        # This can't be a normal validator because it depends on a value
        if self.response is not None and self.response not in self.response_options:
            raise ValidationError(_('%(value)s is not included in %(options)s'),
                params={'value': self.response, 'options': self.response_options}, )
