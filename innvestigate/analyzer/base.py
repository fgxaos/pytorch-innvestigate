import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = [
    "NotAnalyzeableModelException",
    "AnalyzerBase",
    "TrainerMixin",
    "OneEpochTrainerMixin",
    "AnalyzerNetworkBase",
    "ReverseAnalyzerBase"
]

class NotAnalyzeableModelException(Exception):
    """
    Indicates that the model cannot be analyzed by an analyzer
    """
    pass

class AnalyzerBase(object):
    """
    The basic interface of an iNNvestigate analyzer

    This class defines the basic interface for analyzers:
    
    :param model: A PyTorch model
    :param disable_model_checks: Do not execute model checks that enforce
        compatibility of analyzer and model.
    
    .. note:: To develop a new analyzer derive from
        :class:`AnalyzerNetworkBase`
    """

    def __init__(self, model, disable_model_checks=False):
        self._model = model
        self._disable_model_checks = disable_model_checks

        self._do_model_checks()

    def _add_model_checks(self, check, message, check_type="exception"):
        if getattr(self, "_model_check_done", False):
            raise Exception("Cannot add model check anymore."
                            "Check was already performed.")
        
        if not hasattr(self, "_model_checks"):
            self._model_checks = []

        check_instance = {
            "check": check,
            "message": message,
            "type": check_type,
        }
        self._model_checks.append(check_instance)

    def _do_model_checks(self):
        model_checks = getattr(self, "_model_checks", [])

        if not self._disable_model_checks and len(model_checks) > 0:
            check = [x["check"] for x in model_checks]
            types = [x["type"] for x in model_checks]
            messages = [x["message"] for x in model_checks]

            checked = kgraph.model_contains(self._model, check)
            tmp = zip(iutils.to_list(checked), messages, types)

            for checked_layers, message, check_type in tmp:
                if len(checked_layers) > 0:
                    tmp_message = ("%s\nCheck triggered by layers: %s" %
                                    (message, checked_layers))
                    
                    if check_type == "exception":
                        raise NotAnalyzeableModelException(tmp_message)
                    elif check_type == "warning":
                        # TODO(albermax) only the first warning will be shown
                        warnings.warn(tmp_message)
                    else:
                        raise NotImplementedError()

        self._model_checks_done = True

    def fit(self, *args, **kwargs):
        """
        Stub that eats arguments. If an analyzer need training
        include :class:`TrainerMixin`

        :param disable_no_training_warning: Do not warn if this function is
            called despite no training is needed
        """
        disable_no_training_warning = kwargs.pop("disable_no_training_warning", 
                                                False)
        if not disable_no_training_warning:
            # Issue warning if not training is foreseen
            # But is fit is still called
            warnings.warn("This analyzer does not need to be trained."
                        " Still fit() is called.", RuntimeWarning)

    def fit_generator(self, *args, **kwargs):
        """
        Stub that eats arguments. If an analyzer needs training
        include :class:`TrainerMixin`.

        :param disable_no_training_warning: Do not warn if this function is
            called despite no training is needed
        """
        disable_no_training_warning = kwargs.pop("disable_no_training_warning", 
                                                False)
        if not disable_no_training_warning:
            # Issue warning if not training is foreseen,
            # But is fit is still called
            warnings.warn("This analyzer does not need to be trained."
                        " Still fit_generator() is called.", RuntimeWarning)

    def analyze(self, X):
        """
        Analyze the behavior of model on input `X`

        :param X: Input as expected by model
        """
        raise NotImplementedError()
    
    def _get_state(self):
        state = {
            "model_json": self._model.to_json(),
            "model_weights": self._model.get_weights(),
            "disable_model_checks": self._disable_model_checks,
        }
        return state
    
    def save(self):
        """
        Save state of analyzer, can be passed to :func:`Analyzer.load`
        to resemble the analyzer

        :return: The class name and the state
        """
        state = self._get_state()
        class_name = self.__class__.__name__
        return class_name, state

    def save_npz(self, fname):
        """
        Save state of analyzer, can be passed to :func:`Analyzer.load_npz`
        to resemble the analyzer.

        :param fname: The file's name
        """
        class_name, state = self.save()
        np.savez(fname, **{
            "class_name": class_name,
            "state": state
        })
    
    """
    @classmethod
    def _state_to_kwargs(clazz, state):
        model_file = state.pop("model_file")
        model_weights = state.pop("model_weights")
        disable_model_checks = state.pop("disable_model_checks")
        assert len(state) == 0

        model = torch.load
    """

    @staticmethod
    def load(class_name, state):
        """
        Resembles an analyzer from the state created by 
        :func:`analyzer.save()`

        :param class_name: The analyzer's class name
        :param state: The analyzer's state
        """