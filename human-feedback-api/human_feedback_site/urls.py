from django.conf.urls import include, url

from django.contrib import admin
admin.autodiscover()

import human_feedback_api.views

# Examples:
# url(r'^$', 'human_comparison_site.views.home', name='home'),
# url(r'^blog/', include('blog.urls')),

urlpatterns = [
    url(r'^$', human_feedback_api.views.index, name='index'),
    url(r'^experiments/(.*)/list$', human_feedback_api.views.list_comparisons, name='list'),
    url(r'^comparisons/(.*)$', human_feedback_api.views.show_comparison, name='show_comparison'),
    url(r'^experiments/(.*)/ajax_response', human_feedback_api.views.ajax_response, name='ajax_response'),
    url(r'^experiments/(.*)$', human_feedback_api.views.respond, name='responses'),
    url(r'^admin/', include(admin.site.urls)),
]
