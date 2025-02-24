import os
import shutil

import pytest
from ray import tune

from autogluon.core.hpo.ray_tune_constants import SCHEDULER_PRESETS, SEARCHER_PRESETS
from autogluon.multimodal import MultiModalPredictor

from ..others.test_matcher import verify_matcher_save_load
from ..predictor.test_predictor import verify_predictor_save_load
from ..utils.unittest_datasets import IDChangeDetectionDataset, PetFinderDataset
from ..utils.utils import get_home_dir


@pytest.mark.parametrize("searcher", list(SEARCHER_PRESETS.keys()))
@pytest.mark.parametrize("scheduler", list(SCHEDULER_PRESETS.keys()))
def test_predictor_hpo(searcher, scheduler):
    dataset = PetFinderDataset()

    hyperparameters = {
        "optimization.learning_rate": tune.uniform(0.0001, 0.01),
        "optimization.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "fusion_mlp"],
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    hyperparameter_tune_kwargs = {
        "searcher": searcher,
        "scheduler": scheduler,
        "num_trials": 2,
    }

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    save_path = os.path.join(get_home_dir(), "hpo", f"_{searcher}", f"_{scheduler}")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor = predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=save_path,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )

    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df)

    # test for continuous training
    predictor = predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=30,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )


@pytest.mark.parametrize("searcher", list(SEARCHER_PRESETS.keys()))
@pytest.mark.parametrize("scheduler", list(SCHEDULER_PRESETS.keys()))
def test_matcher_hpo(searcher, scheduler):
    dataset = IDChangeDetectionDataset()

    hyperparameters = {
        "optimization.learning_rate": tune.uniform(0.0001, 0.001),
        "optimization.max_epochs": 1,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "optimization.top_k_average_method": "greedy_soup",
    }

    hyperparameter_tune_kwargs = {
        "searcher": searcher,
        "scheduler": scheduler,
        "num_trials": 2,
    }

    matcher = MultiModalPredictor(
        query="Previous Image",
        response="Current Image",
        problem_type="image_similarity",
        label=dataset.label_columns[0] if dataset.label_columns else None,
        match_label=dataset.match_label,
        eval_metric=dataset.metric,
    )

    save_path = os.path.join(get_home_dir(), "hpo", f"_{searcher}", f"_{scheduler}")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    matcher = matcher.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=60,
        save_path=save_path,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )

    score = matcher.evaluate(dataset.test_df)
    verify_matcher_save_load(matcher, dataset.test_df)

    # test for continuous training
    predictor = matcher.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=60,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )


@pytest.mark.parametrize("searcher", list(SEARCHER_PRESETS.keys()))
@pytest.mark.parametrize("scheduler", list(SCHEDULER_PRESETS.keys()))
def test_hpo_distillation(searcher, scheduler):
    dataset = PetFinderDataset()

    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "fusion_mlp"],
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    hyperparameter_tune_kwargs = {
        "searcher": searcher,
        "scheduler": scheduler,
        "num_trials": 2,
    }

    teacher_predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    teacher_save_path = os.path.join(get_home_dir(), "hpo_distillation_teacher", f"_{searcher}", f"_{scheduler}")
    if os.path.exists(teacher_save_path):
        shutil.rmtree(teacher_save_path)

    teacher_predictor = teacher_predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=teacher_save_path,
    )

    hyperparameters = {
        "optimization.learning_rate": tune.uniform(0.0001, 0.01),
        "optimization.max_epochs": 1,
        "model.names": ["numerical_mlp"],
        "data.numerical.convert_to_text": False,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    # test for distillation
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    student_save_path = os.path.join(get_home_dir(), "hpo_distillation_student", f"_{searcher}", f"_{scheduler}")
    if os.path.exists(student_save_path):
        shutil.rmtree(student_save_path)

    predictor = predictor.fit(
        train_data=dataset.train_df,
        teacher_predictor=teacher_save_path,
        hyperparameters=hyperparameters,
        time_limit=60,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        save_path=student_save_path,
    )
