import argparse
import warnings

warnings.filterwarnings('ignore')

from Experiment import dataset_configuration, ForgeryDetectionExperiment
from utils import get_save_path, save_metrics


def run_inference(args):
    print("Hello! We start experiment for Forgery Detection!")

    args = dataset_configuration(args)
    print("Training Arguments : {}".format(args))

    experiment = ForgeryDetectionExperiment(args)
    test_results, total_metrics_dataframe = experiment.fit()

    model_dirs = get_save_path(args)
    print("Save {} Model Test Results...".format(args.model_name))
    save_metrics(args, test_results, model_dirs, total_metrics_dataframe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference-only runner for SARIF.')

    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--train_data_type', type=str, default='dataset')
    parser.add_argument('--save_path', type=str, default='./model_weights')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--efficiency_analysis', '-ea', default=False, action='store_true')
    parser.add_argument('--inference_time', '-ms', default=False, action='store_true')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_L.pth')

    args = parser.parse_args()

    args.model_name = 'SARIF'
    args.train = False

    num_fold = 5
    test_data_type_list = ['CASIAv1']
    for current_fold in range(1, num_fold + 1):
        print(f'\n============ FOLD {current_fold}/{num_fold} ============\n')
        for test_data_type in test_data_type_list:
            args.current_fold = current_fold
            args.test_data_type = test_data_type
            run_inference(args)
