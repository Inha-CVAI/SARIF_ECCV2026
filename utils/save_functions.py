import os

import numpy as np
import torch

import torch.distributed as dist
from .get_functions import get_save_path
__all__ = ['is_main_process', 'save_result', 'save_model','save_metrics']

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def fold_suffix(args):
    if hasattr(args, 'current_fold'):
        return 'fold{}'.format(args.current_fold)
    return getattr(args, 'robustness', 'None')

def save_result(args, model, optimizer, test_results, total_metrics_dataframe):
    if not is_main_process():
        return
    model_dirs = get_save_path(args)

    print("Your experiment is saved in {}.".format(model_dirs))

    print("STEP1. Save {} Model Weight...".format(args.model_name))
    save_model(args, model, optimizer, model_dirs)

    print("STEP2. Save {} Model Test Results...".format(args.model_name))
    save_metrics(args, test_results, model_dirs, total_metrics_dataframe)

    print("EPOCH {} model is successfully saved at {}".format(args.final_epoch, model_dirs))

def save_model(args, model, optimizer, model_dirs, epoch=None):
    if not is_main_process():
        return
    save_epoch = args.final_epoch if epoch is None else epoch
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    check_point = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'current_epoch': save_epoch
    }

    if hasattr(args, 'current_fold'):
        weight_name = 'model_weight_EPOCH{}_fold{}.pth.tar'.format(save_epoch, args.current_fold)
    else:
        weight_name = 'model_weight_EPOCH{}.pth.tar'.format(save_epoch)
    torch.save(check_point, os.path.join(model_dirs, 'model_weights', weight_name))

def save_metrics(args, test_results, model_dirs, total_metrics_dataframe):
    if not is_main_process():
        return
    bootstrap_stats = bootstrap_metric_statistics(args, test_results, total_metrics_dataframe)

    print("###################### TEST REPORT ######################")
    for metric in test_results.keys():
        print("Mean {}    :\t {}".format(metric, test_results[metric]))
    if bootstrap_stats:
        print("###################### BOOTSTRAP 95% CI ######################")
        for metric, stats in bootstrap_stats.items():
            print(
                "{} mean {:.6f} | image_std {:.6f} | bootstrap_std {:.6f} | 95% CI [{:.6f}, {:.6f}]".format(
                    metric,
                    stats["mean"],
                    stats["image_std"],
                    stats["bootstrap_std"],
                    stats["ci95_low"],
                    stats["ci95_high"],
                )
            )
    print("###################### TEST REPORT ######################\n")

    os.makedirs(os.path.join(model_dirs, 'new_test_reports', '{}_{}'.format(args.train_data_type, args.test_data_type)), exist_ok=True)
    os.makedirs(os.path.join(model_dirs, 'new_test_reports_per_cases', '{}_{}'.format(args.train_data_type, args.test_data_type)), exist_ok=True)
    os.makedirs(os.path.join(model_dirs, 'new_test_reports_bootstrap', '{}_{}'.format(args.train_data_type, args.test_data_type)), exist_ok=True)

    suffix = fold_suffix(args)
    test_results_save_path = os.path.join(model_dirs, 'new_test_reports', '{}_{}'.format(args.train_data_type, args.test_data_type),
                                          'test_reports_EPOCH{}_{}_{}_{}.txt'.format(args.final_epoch, args.train_data_type, args.test_data_type, suffix))
    test_results_csv_save_path = os.path.join(model_dirs, 'new_test_reports_per_cases',  '{}_{}'.format(args.train_data_type, args.test_data_type),
                                              'test_reports_EPOCH{}_{}_{}_{}.csv'.format(args.final_epoch, args.train_data_type, args.test_data_type, suffix))
    bootstrap_csv_save_path = os.path.join(model_dirs, 'new_test_reports_bootstrap', '{}_{}'.format(args.train_data_type, args.test_data_type),
                                           'bootstrap_EPOCH{}_{}_{}_{}.csv'.format(args.final_epoch, args.train_data_type, args.test_data_type, suffix))

    f = open(test_results_save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    for metric in test_results.keys():
        f.write("Mean {}    :\t {}\n".format(metric, test_results[metric]))
    if bootstrap_stats:
        f.write("###################### BOOTSTRAP 95% CI ######################\n")
        for metric, stats in bootstrap_stats.items():
            f.write(
                "{} mean {:.6f} | image_std {:.6f} | bootstrap_std {:.6f} | 95% CI [{:.6f}, {:.6f}]\n".format(
                    metric,
                    stats["mean"],
                    stats["image_std"],
                    stats["bootstrap_std"],
                    stats["ci95_low"],
                    stats["ci95_high"],
                )
            )
    f.write("###################### TEST REPORT ######################\n")

    f.close()

    print("test results txt file is saved at {}".format(test_results_save_path))

    total_metrics_dataframe.to_csv(test_results_csv_save_path, index=False)
    if bootstrap_stats:
        save_bootstrap_statistics(bootstrap_stats, bootstrap_csv_save_path)


def bootstrap_metric_statistics(args, test_results, total_metrics_dataframe):
    iterations = int(getattr(args, "bootstrap_iterations", 1000))
    if iterations <= 0 or total_metrics_dataframe is None or total_metrics_dataframe.empty:
        return {}

    rng = np.random.default_rng(int(getattr(args, "bootstrap_seed", 4321)))
    stats_by_metric = {}
    for metric in test_results.keys():
        if metric not in total_metrics_dataframe.columns:
            continue

        values = total_metrics_dataframe[metric].dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue

        bootstrap_means = np.empty(iterations, dtype=np.float64)
        for idx in range(iterations):
            sampled = rng.choice(values, size=values.size, replace=True)
            bootstrap_means[idx] = sampled.mean()

        stats_by_metric[metric] = {
            "mean": float(values.mean()),
            "image_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "bootstrap_std": float(bootstrap_means.std(ddof=1)) if iterations > 1 else 0.0,
            "ci95_low": float(np.percentile(bootstrap_means, 2.5)),
            "ci95_high": float(np.percentile(bootstrap_means, 97.5)),
            "num_images": int(values.size),
            "bootstrap_iterations": iterations,
        }

    return stats_by_metric


def save_bootstrap_statistics(bootstrap_stats, save_path):
    header = "metric,mean,image_std,bootstrap_std,ci95_low,ci95_high,num_images,bootstrap_iterations\n"
    with open(save_path, "w") as handle:
        handle.write(header)
        for metric, stats in bootstrap_stats.items():
            handle.write(
                "{},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{},{}\n".format(
                    metric,
                    stats["mean"],
                    stats["image_std"],
                    stats["bootstrap_std"],
                    stats["ci95_low"],
                    stats["ci95_high"],
                    stats["num_images"],
                    stats["bootstrap_iterations"],
                )
            )
