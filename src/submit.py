
## SEED

np.random.seed(42)

## SUBMISSIONS

SUBMISSIONS = {
    'shortterm': {
        '1': 'modelX',
    }
    'longterm': {
        '1': 'modelY',
    }
}
SUBMISSION_NAME = 'runs/me19mem_insightdcu_{}_run{}.csv'

# load test data
# used the predictions generated for test data

for target in SUBMISSIONS.keys():

    for run_number, model_name in SUBMISSIONS[target].items():

        # load model
        # run predictions

