import multiprocessing
import os
import os.path as osp
import uuid

import re
import numpy as np

from django.conf import settings
from human_feedback_api.models import Comparison
from time import time


from rl_teacher.envs import make_with_torque_removed
from rl_teacher.video import write_segment_to_video, upload_to_gcs
from rl_teacher.utils import tprint

class SyntheticComparisonCollector(object):
    def __init__(self):
        self._comparisons = []

    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "label": None
        }
        self._comparisons.append(comparison)

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self):
        for comp in self.unlabeled_comparisons:
            self._add_synthetic_label(comp)

    @staticmethod
    def _add_synthetic_label(comparison):
        left_seg = comparison['left']
        right_seg = comparison['right']
        left_has_more_rew = np.sum(left_seg["original_rewards"]) > np.sum(right_seg["original_rewards"])

        # Mutate the comparison and give it the new label
        comparison['label'] = 0 if left_has_more_rew else 1

def _write_and_upload_video(env_id, gcs_path, local_path, segment):
    env = make_with_torque_removed(env_id)
    write_segment_to_video(segment, fname=local_path, env=env)
    serve_locally = settings.MEDIA_URL == '/media/'
    if serve_locally or not gcs_path:
        found = re.search(r'\/(\w{8}\-)',local_path)
        tprint("media_url OK, path=",local_path, "             >>> ", found.group())
    else:
        upload_to_gcs(local_path, gcs_path)
        found = re.search(r'\/(\w{8}\-)',gcs_path)
        tprint("GCS upload OK, path=",gcs_path, "             >>> ", found.group())

class HumanComparisonCollector():
    def __init__(self, env_id, experiment_name, workers=4, learn_predictor=True, reset=False, force=False):
        from human_feedback_api import Comparison

        self._comparisons = []
        self.env_id = env_id
        self.experiment_name = experiment_name
        self.workers = workers
        self._upload_workers = multiprocessing.Pool(self.workers)
        self.learn_predictor =  learn_predictor
        self.wait_until = time()

        if Comparison.objects.filter(experiment_name=experiment_name).count() > 0:
            if reset:
                # delete existing experiment
                Comparison.objects.filter(experiment_name=experiment_name).delete()
            elif not force:
                raise EnvironmentError("Existing experiment named %s! Pick a new experiment name." % experiment_name)

    @staticmethod
    def _add_synthetic_label(comparison):
        left_seg = comparison['left']
        right_seg = comparison['right']
        left_has_more_rew = np.sum(left_seg["original_rewards"]) > np.sum(right_seg["original_rewards"])

        # Mutate the comparison and give it the new label
        comparison['label'] = 0 if left_has_more_rew else 1

    def convert_segment_to_media_url(self, comparison_uuid, side, segment):
        # NOTE: make sure this matches settings.MEDIA_ROOT
        # see: rl-teacher/human-feedback-api/human_feedback_site/settings.py

        tmp_media_dir = settings.MEDIA_ROOT + self.experiment_name
        media_id = "%s-%s.mp4" % (comparison_uuid, side)
        local_path = osp.join(tmp_media_dir, media_id)
        serve_locally = settings.MEDIA_URL == '/media/'
        if serve_locally:
            found = re.search(r'\/(\w{8}\-)',local_path)
            tprint("## write> path=",local_path, "         write >>> ", found.group())
            result = self._upload_workers.apply_async(_write_and_upload_video, (self.env_id, None, local_path, segment))
            media_url = "/media/%s/%s" % (self.experiment_name, media_id)
        else:
            # serve segment from GCS
            gcs_bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
            gcs_path = osp.join(gcs_bucket, media_id)
            self._upload_workers.apply_async(_write_and_upload_video, (self.env_id, gcs_path, local_path, segment))
            media_url = "https://storage.googleapis.com/%s/%s" % (gcs_bucket.lstrip("gs://"), media_id)

        return media_url

    def _create_comparison_in_webapp(self, left_seg, right_seg, response=None):
        """Creates a comparison DB object. Returns the db_id of the comparison"""
        # from human_feedback_api import Comparison

        comparison_uuid = str(uuid.uuid4())
        comparison = Comparison(
            experiment_name=self.experiment_name,
            media_url_1=self.convert_segment_to_media_url(comparison_uuid, 'left', left_seg),
            media_url_2=self.convert_segment_to_media_url(comparison_uuid, 'right', right_seg),
            response_kind='left_or_right',
            priority=1.
        )
        if not (response is None): comparison["response"]=response
        comparison.full_clean()
        comparison.save()
        return comparison.id

    def add_segment_pair(self, left_seg, right_seg, force_webapp=False):
        ### rate control - trying to avoid 
        #   Break on __THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_FUNCTIONALITY___YOU_MUST_EXEC__() to debug.



        """Add a new unlabeled comparison from a segment pair"""
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "label": None
        }
        use_synthetic_label = not force_webapp and not self.learn_predictor

        # wait 100ms between each call to create webapp comparison
        if not use_synthetic_label and time() < self.wait_until: 
            use_synthetic_label = True
        else:
            self.wait_until = time() + 0.100

        if use_synthetic_label:
            ### using pre-trained predictor, so don't ask for human feeback
            ### Q: Whats the point of adding comparisons? Are they used for TRPO learning
            # i.e. policy or Q network learning? or only for reward predictor learning?
            print(">>> using synthetic labels, comparisons=",len(self._comparisons))
            self._add_synthetic_label(comparison)
            self._comparisons.append(comparison)
        else:
            resp = "checkpoint" if force_webapp else None
            comparison["id"] = self._create_comparison_in_webapp(left_seg, right_seg, resp)
            self._comparisons.append(comparison)

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    # get label from convert human feedback
    def label_unlabeled_comparisons(self):
        for comparison in self.unlabeled_comparisons:
            db_comp = Comparison.objects.get(pk=comparison['id'])
            if db_comp.response == 'left':
                comparison['label'] = 0
            elif db_comp.response == 'right':
                comparison['label'] = 1
            elif db_comp.response == 'tie' or db_comp.response == 'abstain':
                comparison['label'] = 'equal'
                # If we did not match, then there is no response yet, so we just wait
