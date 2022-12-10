from django.conf.urls import include
from django.urls import re_path

from django.contrib import admin
admin.autodiscover()

import human_feedback_api.views

# Examples:
# url(r'^$', 'human_comparison_site.views.home', name='home'),
# url(r'^blog/', include('blog.urls')),

urlpatterns = [
    re_path(r'^$', human_feedback_api.views.index, name='index'),
    re_path(r'^experiments/(.*)/list$', human_feedback_api.views.list_comparisons, name='list'),
    re_path(r'^comparisons/(.*)$', human_feedback_api.views.show_comparison, name='show_comparison'),
    re_path(r'^experiments/(.*)/ajax_response', human_feedback_api.views.ajax_response, name='ajax_response'),
    re_path(r'^experiments/(.*)$', human_feedback_api.views.respond, name='responses'),
    re_path(r'^admin/', include(admin.site.urls[:2])),
]
