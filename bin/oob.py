"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""

import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import label_binarize

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score



import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))  # append the MIALab root directory to Python path
# fixes the ModuleNotFoundError when executing main.py in the console after code changes (e.g. git pull)
# somehow pip install does not keep track of packages

import mialab.data.structure as structure
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil
from sklearn.metrics import f1_score

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load







def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    warnings.warn('Random forest parameters not properly set.')
    # visualization(images)
    print(np.shape(images[0].feature_matrix[0]))

    error_rate=[]
    for num_estimators in range(1,10):
        forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                    n_estimators=num_estimators,
                                                    max_depth=10,oob_score=True)

        # start_time = timeit.default_timer()
        forest.fit(data_train, labels_train)

        oob_error = 1 - forest.oob_score_
        print(forest.oob_score_)
        error_rate+=[oob_error]




    plt.plot(range(1,10),error_rate)
    plt.show()


    #
    # print(' Time elapsed:', timeit.default_timer() - start_time, 's')
    #
    # # create a result directory with timestamp
    # t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # result_dir = os.path.join(result_dir, t)
    #
    # os.makedirs(result_dir, exist_ok=True)
    #
    # print('-' * 5, 'Testing...')
    #
    # # initialize evaluator
    # evaluator = putil.init_evaluator()
    #
    # # crawl the training image directories
    # crawler = futil.FileSystemDataCrawler(data_test_dir,
    #                                       LOADING_KEYS,
    #                                       futil.BrainImageFilePathGenerator(),
    #                                       futil.DataDirectoryFilter())
    #
    # # load images for testing and pre-process
    # pre_process_params['training'] = False
    # images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)
    #
    # data_test = np.concatenate([img.feature_matrix[0] for img in images_test])
    # labels_test = np.concatenate([img.feature_matrix[1] for img in images_test]).squeeze()
    #
    # random_state = np.random.RandomState(0)
    # # ax = plt.gca()
    # # rfc_disp = plot_roc_curve(forest, data_test, labels_test, ax=ax, alpha=0.8)
    # # svc_disp.plot(ax=ax, alpha=0.8)
    # # disp = plot_confusion_matrix(forest, data_test, labels_test, normalize='true')
    # # plt.show()
    # X= np.concatenate((data_train,data_test))
    # y= np.concatenate((labels_train,labels_test))
    # y = label_binarize(y, classes=[0, 1, 2 , 3, 4 , 5])
    # n_classes = y.shape[1]
    # n_samples, n_features = X.shape
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,random_state=0)
    #
    # # classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
    # #                                          random_state=random_state))
    #
    # classifier = OneVsRestClassifier(sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
    #                                             n_estimators=10,
    #                                             max_depth=5))
    #
    # y_score = classifier.fit(X_train, y_train).predict(X_test)
    #
    #
    #
    # # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #
    # # Finally average it and compute AUC
    # mean_tpr /= n_classes
    #
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    # plt.figure()
    # lw = 2
    # plt.plot(fpr[2], tpr[2], color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
    #
    #
    # # Plot all ROC curves
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)
    #
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()

    # evaluator.clear()


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
