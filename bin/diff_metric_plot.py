"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""

import argparse
import datetime
import os
import sys
import timeit
import warnings
import pandas as pd
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import label_binarize


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
    dfs= []
    aggregated_results = []

    print('-' * 5, 'Testing...')
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)


    for num_estimator in [10]:
        forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                    n_estimators=num_estimator,
                                                    max_depth=10)

        start_time = timeit.default_timer()
        forest.fit(data_train, labels_train)

        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # create a result directory with timestamp
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        result_dir = os.path.join(result_dir, t)

        os.makedirs(result_dir, exist_ok=True)

        print('-' * 5, 'Testing...')

        # initialize evaluator
        evaluator = putil.init_evaluator()

        # crawl the training image directories
        # crawler = futil.FileSystemDataCrawler(data_test_dir,
        #                                       LOADING_KEYS,
        #                                       futil.BrainImageFilePathGenerator(),
        #                                       futil.DataDirectoryFilter())

        # load images for testing and pre-process

        # data_test = np.concatenate([img.feature_matrix[0] for img in images_test])
        # labels_test = np.concatenate([img.feature_matrix[1] for img in images_test]).squeeze()

        # ax = plt.gca()
        # rfc_disp = plot_roc_curve(forest, data_test, labels_test, ax=ax, alpha=0.8)
        # svc_disp.plot(ax=ax, alpha=0.8)
        # disp = plot_confusion_matrix(forest, data_test, labels_test, normalize='true')
        # plt.show()

        # y = label_binarize(labels_test, classes=[0, 1, 2 , 3, 4 , 5])
        # n_classes = y.shape[1]


        images_prediction = []
        images_probabilities = []


        for img in images_test:
            print('-' * 10, 'Testing', img.id_)


            start_time = timeit.default_timer()
            predictions = forest.predict(img.feature_matrix[0])
            probabilities = forest.predict_proba(img.feature_matrix[0])
            print(' Time elapsed:', timeit.default_timer() - start_time, 's')

            # convert prediction and probabilities back to SimpleITK images
            image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                            img.image_properties)
            image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

            # evaluate segmentation without post-processing
            evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

            images_prediction.append(image_prediction)
            images_probabilities.append(image_probabilities)

        results=evaluator.results
        labels = sorted({result.label for result in results})
        metrics = sorted({result.metric for result in results})

        # functions = {'MEAN': np.mean, 'STD': np.std}
        functions = {'MEAN': np.mean}
        for label in labels:
            for metric in metrics:
                # search for results
                values = [r.value for r in results if r.label == label and r.metric == metric]


                for fn_id, fn in functions.items():
                    aggregated_results.append(
                        [num_estimator,
                        label,
                        metric,
                        float(fn(values))])

        # for result in aggregated_results:
        #     # print([result.label, result.metric, result.id_, result.value])
        #     print(result)


        # writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

        # clear results such that the evaluator is ready for the next evaluation
        evaluator.clear()
    df=pd.DataFrame(aggregated_results, columns=['n_estimators', 'label', 'metric', 'value'])
    return df
    xdf = df[df.label == 'WhiteMatter']
    del xdf['label']




    # new_df=df[df.label=='GreyMatter']
    # del new_df['label']
    # new_df.set_index('n_estimators', inplace=True)
    # fig, ax = plt.subplots(figsize=(15, 7))
    # new_df.groupby(['metric']).plot(ax=ax)
    # print(new_df)

    # plt.show()

    plt.figure(2)
    # pd.crosstab(index=[df['Name'], df['Date']], columns=new_df['metric'])
    my_df = pd.pivot_table(df,index=['label'], columns='metric', values='value')
    my_df.plot()
    print(my_df)




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
    df=main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
