# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.return_feats = cfg.MODEL.ROI_BOX_HEAD.RETURN_FC_FEATS
        self.has_attributes = cfg.MODEL.ROI_BOX_HEAD.ATTR

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        if self.return_feats:
            if self.has_attributes:
                out_dict = self.predictor(x, proposals)
                attr_logits = out_dict["attr_score"]
            else:
                out_dict = self.predictor(x)

            class_logits = out_dict["scores"]
            box_regression = out_dict["bbox_deltas"]
        else:
            class_logits, box_regression = self.predictor(x)

        if not self.training:
            # no need of post processing
            # the bounding box is ground truth and that's enough
            #result = self.post_processor((class_logits, box_regression), proposals)
            result = proposals
            if self.return_feats:
                return out_dict, result, {}
            else:
                return x, result, {}

        if self.has_attributes:
            loss_classifier, loss_box_reg, loss_attr = self.loss_evaluator(
                [class_logits], [box_regression], [attr_logits]
            )
            return (
                x,
                proposals,
                dict(
                    loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_attr=loss_attr
                ),
            )

        else:
            loss_classifier, loss_box_reg = self.loss_evaluator(
                [class_logits], [box_regression], [attr_logits]
            )

            return (x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg))


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
