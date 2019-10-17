# imports
from scipy.stats import spearmanr


def evaluate_spearman(true, predicted):
    """ Evaluate using Spearman correlation """

    return spearmanr(true, predicted)


if __name__ == "__main__":
    corr_coefficient, p_value = evaluate_spearman(
        [1, 2, 3, 4, 5],
        [1, 4, 9, 26, 25],
    )
    print('Spearman Correlation Coefficient: {:.4f} (p-value {:.4f})'.format(corr_coefficient, p_value))