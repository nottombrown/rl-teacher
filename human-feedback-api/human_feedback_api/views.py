from collections import namedtuple
from datetime import timedelta, datetime

from django import template
from django.db.models import Q
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.utils import timezone
from django.http import HttpResponse
import pytz
import json

from human_feedback_api.models import Comparison
import re

register = template.Library()

ExperimentResource = namedtuple("ExperimentResource", ['name', 'num_responses', 'started_at', 'pretty_time_elapsed'])

def _pretty_time_elapsed(start, end):
    total_seconds = (end - start).total_seconds()
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))

def _build_experiment_resource(experiment_name):
    comparisons = Comparison.objects.filter(experiment_name=experiment_name, responded_at__isnull=False)
    try:
        started_at = comparisons.order_by('-created_at').first().created_at
        pretty_time_elapsed = _pretty_time_elapsed(started_at, timezone.now())
    except AttributeError:
        started_at = None
        pretty_time_elapsed = None
    return ExperimentResource(
        name=experiment_name,
        num_responses=comparisons.count(),
        started_at=started_at,
        pretty_time_elapsed=pretty_time_elapsed
    )

def _all_comparisons(experiment_name, use_locking=True):
    not_responded = Q(responded_at__isnull=True)

    cutoff_time = timezone.now() - timedelta(minutes=5)
    not_in_progress = Q(shown_to_tasker_at__isnull=True) | Q(shown_to_tasker_at__lte=cutoff_time)
    finished_uploading_media = Q(created_at__lte=timezone.now() - timedelta(seconds=25)) # Give time for upload
    ready = not_responded & not_in_progress & finished_uploading_media
    
    # if not ready:
    #     print("NOT ready", cutoff_time, not_in_progress, finished_uploading_media)  

    #
    # BUG: (pretraining) getting the last comparisons first (created_at DESC), but videos created by created_at ASC
    #       during learning, we want to offer feedback on the most recent samples (ones with most learning)
    #
    #  OR, create pre-training video samples in the same order, most _recent
    #
    ## pretraining phase
    isPretraining=True
    if isPretraining:
        ## Sort by priority, then put OLDEST labels first
        ready = not_responded & finished_uploading_media
        return Comparison.objects.filter(ready, experiment_name=experiment_name).order_by('-priority', 'id')
    else:
        ## learning phase  
        # Sort by priority, then put newest labels first
        return Comparison.objects.filter(ready, experiment_name=experiment_name).order_by('-priority', '-created_at')

def index(request):
    return render(request, 'index.html', context=dict(
        experiment_names=[exp for exp in
            Comparison.objects.order_by().values_list('experiment_name', flat=True).distinct()]
    ))

def list_comparisons(request, experiment_name):
    # comparisons = Comparison.objects.filter(experiment_name=experiment_name).order_by('responded_at', '-priority')
    comparisons = Comparison.objects.filter(experiment_name=experiment_name).order_by('responded_at', '-priority', 'created_at', 'id')
    for c in comparisons: c.uuid=c.media_url_1.split('/')[4]
    return render(request, 'list.html', context=dict(comparisons=comparisons, experiment_name=experiment_name))

def display_comparison(comparison):
    """Mark comparison as having been displayed"""
    comparison.shown_to_tasker_at = timezone.now()
    comparison.save()

def ajax_response(request, experiment_name):
    """Update a comparison with a response"""

    POST = request.POST
    comparison_id = POST.get("comparison_id")
    debug = True

    comparison = Comparison.objects.get(pk=comparison_id)

    # Update the values
    comparison.response = POST.get("response")
    comparison.responded_at = timezone.now()

    if debug:
        print("Answered comparison {} with {}".format(comparison_id, comparison.response))

    comparison.full_clean()  # Validation
    comparison.save()

    limit = 1
    comparisons = list(_all_comparisons(experiment_name)[:limit])
    for comparison in comparisons: display_comparison(comparison)
    if debug:
        print("{}".format([x.id for x in comparisons]))
        if comparison:
            print("Requested {}".format(comparison.id))
    return render(request, 'ajax_response.html', context={
        'comparisons': comparisons,
        'experiment': _build_experiment_resource(experiment_name)
    })

def show_comparison(request, comparison_id):
    comparison = get_object_or_404(Comparison, pk=comparison_id)
    found = re.search(r'\/(\w{8}\-)',comparison.media_url_1)
    return render(request, 'show_feedback.html', context={"feedback": comparison, "uuid": found.group() if found else ''})


def respond(request, experiment_name):
    limit = 20
    comparisons = list(_all_comparisons(experiment_name))[:limit]  
    for comparison in comparisons:
        display_comparison(comparison)
        
    return render(request, 'responses.html', context={
        'comparisons': comparisons,
        'experiment': _build_experiment_resource(experiment_name)
    })

def get_experiment(request, experiment_name, comparison_id):
    comparison = get_object_or_404(Comparison, pk=comparison_id)
    return render(request, 'one_response.html', context={
        'comparison': comparison,
        'experiment': _build_experiment_resource(experiment_name)
    })

def ajax_one_response(request, experiment_name):
    """Update a comparison with just one response"""

    POST = request.POST
    comparison_id = POST.get("comparison_id")
    debug = True

    comparison = Comparison.objects.get(pk=comparison_id)

    # Update the values
    comparison.response = POST.get("response")
    comparison.responded_at = timezone.now()

    if debug:
        print("Answered comparison {} with {}".format(comparison_id, comparison.response))

    comparison.full_clean()  # Validation
    comparison.save()
    payload = {"comparison_id":comparison.id, "response":comparison.response}
    return HttpResponse(json.dumps(payload), content_type="application/json")
