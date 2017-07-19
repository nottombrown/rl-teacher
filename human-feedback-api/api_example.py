from human_feedback_api import Comparison

def generate_test():
    # TEST_MEDIA = ['test707470989.png', 'test559628392.png', 'test736369680.png', 'test836374824.png',
    #     'test593516009.png', 'test836986396.png', 'test996736825.png', 'test746178624.png', 'test58579485.png', ]
    TEST_MEDIA = ['https://storage.googleapis.com/rl-teacher-tom/test_obs_75.mp4'] * 5

    for media_1 in TEST_MEDIA:
        for media_2 in TEST_MEDIA:
            test_comparison = Comparison(
                experiment_name='test_experiment',
                media_url_1=media_1,
                media_url_2=media_2,
                response_kind='left_or_right',
                priority=1.)
            print('Creating test comparison: {}'.format(test_comparison))

            test_comparison.full_clean()
            test_comparison.save()
            import ipdb; ipdb.set_trace()

    print('We now have a total of {} comparisons in the database'.format(Comparison.objects.count()))

if __name__ == '__main__':
    Comparison.objects.all().delete()
    generate_test()

    for comparison in Comparison.objects.filter():
        print("comparison %s => %s" % (comparison.id, comparison.response))
