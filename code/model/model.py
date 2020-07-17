import torch
import torch.nn as nn
from model.resnet3d_xl import Net
import torch.nn.functional as F
'''
Video Classification Model library.
'''

class TrainingScheduleError(Exception):
    pass

class VideoModel(nn.Module):
    def __init__(self,
                 num_classes,
                 num_boxes,
                 num_videos=16,
                 restore_dict=None,
                 freeze_weights=None,
                 device=None,
                 loss_type='softmax'):
        super(VideoModel, self).__init__()
        self.device = device
        self.num_frames = num_videos
        self.num_classes = num_classes
        # Network loads kinetic pre-trained weights in initialization
        self.i3D = Net(num_classes, extract_features=True, loss_type=loss_type)


        try:
            # Restore weights
            if restore_dict:
                self.restore(restore_dict)
            # Freeze weights
            if freeze_weights:
                self.freeze_weights(freeze_weights)
            else:
                print(" > No weights are freezed")
        except Exception as e:
            print(" > Exception {}".format(e))

    def restore(self, restore=None):
        # Load pre-trained I3D + Graph weights for fine-tune (replace the last FC)
        restore_finetuned = restore.get("restore_finetuned", None)
        if restore_finetuned:
            self._restore_fintuned(restore_finetuned)
            print(" > Restored I3D + Graph weights")
            return

        # Load pre-trained I3D weights
        restore_i3d = restore.get("restore_i3d", None)
        if restore_i3d:
            self._restore_i3d(restore_i3d)
            print(" > Restored only I3D weights")
            return

        # Load pre-trained I3D + Graph weights without replacing anything
        restore_predict = restore.get("restore_predict", None)
        if restore_predict:
            self._restore_predict(restore_predict)
            print(" > Restored the model with strict weights")
            return

    def _restore_predict(self, path):
        if path is None:
            raise TrainingScheduleError('You should pre-train the video model on your training data first')

        weights = torch.load(path, map_location=self.device)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            new_weights[k.replace('module.', '')] = v

        self.load_state_dict(new_weights, strict=True)
        print(" > Weights {} loaded".format(path))

    def _restore_i3d(self, path):
        if path is None:
            raise TrainingScheduleError('You should pre-train the video model on your training data first')
       
        weights = torch.load(path, map_location=self.device)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not k.startswith('module.fc') and not k.startswith('module.i3D.classifier'):
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)

    def _restore_fintuned(self, path):
        if path is None:
            raise TrainingScheduleError('You should pre-train the video model on your training data first')

        weights = torch.load(path, map_location=self.device)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            # Don't load classifiers (different classes 88 vs 86)
            if not k.startswith('module.fc'):
                if not k.startswith('module.i3D.classifier'):
                    new_weights[k.replace('module.', '')] = v

        self.load_state_dict(new_weights, strict=False)
        print(" > Weights {} loaded".format(path))

    def freeze_weights(self, module):
        if module == 'i3d':
            print(" > Freeze I3D module")
            for param in self.i3D.parameters():
                param.requires_grad = False
        elif module == 'fine_tuned':
            print(" > Freeze Graph + I3D module, only last FC is training")
            # Fixed the entire params without the last FC
            for name, param in self.i3D.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False
            for param in self.graph_embedding.parameters():
                param.requires_grad = False
            for param in self.conv.parameters():
                param.requires_grad = False

        else:
            raise NotImplementedError('Unrecognized option, you can freeze either graph module or I3D module')
        pass

    def _get_i3d_features(self, videos, output_video_features=False):
        # org_features - [V x 2048 x T / 2 x 14 x 14]
        _, org_features = self.i3D(videos)
        # Reduce dimension video_features - [V x 512 x T / 2 x 14 x 14]
        videos_features = self.conv(org_features)
        bs, d, t, h, w = videos_features.size()
        # Get global features
        videos_features_rs = videos_features.permute(0, 2, 1, 3, 4)  # [V x T / 2 x 512 x h x w]
        videos_features_rs = videos_features_rs.reshape(-1, d, h, w)  # [V * T / 2 x 512 x h x w]
        global_features = self.avgpool(videos_features_rs)  # [V * T / 2 x 512 x 1 x 1]
        global_features = self.dropout(global_features)
        global_features = global_features.reshape(bs, t, d)  # [V x T / 2 x 512]
        if output_video_features:
            return global_features, videos_features
        else:
            return global_features

    def flatten(self, x):
        return [item for sublist in x for item in sublist]

