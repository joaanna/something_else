import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet3d_xl import Net
from model.nonlocal_helper import Nonlocal


class VideoModelCoord(nn.Module):
    def __init__(self, opt):
        super(VideoModelCoord, self).__init__()
        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames // 2
        self.coord_feature_dim = opt.coord_feature_dim

        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            # nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512), #self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        #import pdb
        for k, v in weights.items():
            if not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        #pdb.set_trace()
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:

                param.requires_grad = False
                frozen_weights += 1

            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, box_categories, box_input, video_label, is_inference=False):
        # local_img_tensor is (b, nr_frames, nr_boxes, 3, h, w)
        # global_img_tensor is (b, nr_frames, 3, h, w)
        # box_input is (b, nr_frames, nr_boxes, 4)

        b, _, _, _h, _w = global_img_input.size()
        # global_imgs = global_img_input.view(b*self.nr_frames, 3, _h, _w)
        # local_imgs = local_img_input.view(b*self.nr_frames*self.nr_boxes, 3, _h, _w)

        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b*self.nr_boxes*self.nr_frames, 4)

        bf = self.coord_to_feature(box_input)
        bf = bf.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)

        # spatial message passing (graph)
        spatial_message = bf.sum(dim=1, keepdim=True)  # (b, 1, self.nr_frames, coord_feature_dim)
        # message passed should substract itself, and normalize to it as a single feature
        spatial_message = (spatial_message - bf) / (self.nr_boxes - 1)  # message passed should substract itself
        bf_and_message = torch.cat([bf, spatial_message], dim=3)  # (b, nr_boxes, nr_frames, 2*coord_feature_dim)

        # (b*nr_boxes*nr_frames, coord_feature_dim)
        bf_spatial = self.spatial_node_fusion(bf_and_message.view(b*self.nr_boxes*self.nr_frames, -1))
        bf_spatial = bf_spatial.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)

        bf_temporal_input = bf_spatial.view(b, self.nr_boxes, self.nr_frames*self.coord_feature_dim)

        box_features = self.box_feature_fusion(bf_temporal_input.view(b*self.nr_boxes, -1))  # (b*nr_boxes, coord_feature_dim)
        box_features = torch.mean(box_features.view(b, self.nr_boxes, -1), dim=1)  # (b, coord_feature_dim)
        # video_features = torch.cat([global_features, local_features, box_features], dim=1)
        video_features = box_features

        cls_output = self.classifier(video_features)  # (b, num_classes)
        return cls_output

class VideoModelCoordLatent(nn.Module):
    def __init__(self, opt):
        super(VideoModelCoordLatent, self).__init__()
        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames // 2
        self.img_feature_dim = opt.img_feature_dim
        self.coord_feature_dim = opt.coord_feature_dim

        self.category_embed_layer = nn.Embedding(3, opt.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)

        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            # nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512), #self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, box_categories, box_input, video_label, is_inference=False):
        # local_img_tensor is (b, nr_frames, nr_boxes, 3, h, w)
        # global_img_tensor is (b, nr_frames, 3, h, w)
        # box_input is (b, nr_frames, nr_boxes, 4)

        b, _, _, _h, _w = global_img_input.size()

        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b*self.nr_boxes*self.nr_frames, 4)

        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(b*self.nr_boxes*self.nr_frames)
        box_category_embeddings = self.category_embed_layer(box_categories)  # (b*nr_b*nr_f, coord_feature_dim//2)

        bf = self.coord_to_feature(box_input)
        bf = torch.cat([bf, box_category_embeddings], dim=1)  # (b*nr_b*nr_f, coord_feature_dim + coord_feature_dim//2)
        bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)
        bf = bf.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)

        # spatial message passing (graph)
        spatial_message = bf.sum(dim=1, keepdim=True)  # (b, 1, self.nr_frames, coord_feature_dim)
        # message passed should substract itself, and normalize to it as a single feature
        spatial_message = (spatial_message - bf) / (self.nr_boxes - 1)  # message passed should substract itself
        bf_and_message = torch.cat([bf, spatial_message], dim=3)  # (b, nr_boxes, nr_frames, 2*coord_feature_dim)

        # (b*nr_boxes*nr_frames, coord_feature_dim)
        bf_spatial = self.spatial_node_fusion(bf_and_message.view(b*self.nr_boxes*self.nr_frames, -1))
        bf_spatial = bf_spatial.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)

        bf_temporal_input = bf_spatial.view(b, self.nr_boxes, self.nr_frames*self.coord_feature_dim)

        box_features = self.box_feature_fusion(bf_temporal_input.view(b*self.nr_boxes, -1))  # (b*nr_boxes, coord_feature_dim)
        box_features = torch.mean(box_features.view(b, self.nr_boxes, -1), dim=1)  # (b, coord_feature_dim)
        # video_features = torch.cat([global_features, local_features, box_features], dim=1)
        video_features = box_features

        cls_output = self.classifier(video_features)  # (b, num_classes)
        return cls_output

class VideoModelCoordLatentNL(nn.Module):
    def __init__(self, opt):
        super(VideoModelCoordLatentNL, self).__init__()
        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames // 2
        self.img_feature_dim = opt.img_feature_dim
        self.coord_feature_dim = opt.coord_feature_dim

        self.category_embed_layer = nn.Embedding(3, opt.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)

        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim // 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim + self.coord_feature_dim // 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim * 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.nr_nonlocal_layers = 3
        self.nonlocal_fusion = []
        for i in range(self.nr_nonlocal_layers):
            self.nonlocal_fusion.append(nn.Sequential(
                Nonlocal(dim=self.coord_feature_dim, dim_inner=self.coord_feature_dim // 2),
                nn.Conv1d(self.coord_feature_dim, self.coord_feature_dim, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm1d(self.coord_feature_dim),
                nn.ReLU()
            ))
        self.nonlocal_fusion = nn.ModuleList(self.nonlocal_fusion)

        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            # nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),  # self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)

    def train(self, mode=True):  # overriding default train function
        super(VideoModelCoordLatentNL, self).train(mode)
        for m in self.modules():  # or self.modules(), if freezing all bn layers
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        # import pdb
        for k, v in weights.items():
            if not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        # pdb.set_trace()
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:

                param.requires_grad = False
                frozen_weights += 1

            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, box_categories, box_input, video_label, is_inference=False):
        # local_img_tensor is (b, nr_frames, nr_boxes, 3, h, w)
        # global_img_tensor is (b, nr_frames, 3, h, w)
        # box_input is (b, nr_frames, nr_boxes, 4)

        b, _, _, _h, _w = global_img_input.size()

        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b * self.nr_boxes * self.nr_frames, 4)

        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(b * self.nr_boxes * self.nr_frames)
        box_category_embeddings = self.category_embed_layer(box_categories)  # (b*nr_b*nr_f, coord_feature_dim//2)

        bf = self.coord_to_feature(box_input)
        bf = torch.cat([bf, box_category_embeddings], dim=1)  # (b*nr_b*nr_f, coord_feature_dim + coord_feature_dim//2)
        bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)
        bf = bf.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)

        # spatial message passing (graph)
        spatial_message = bf.sum(dim=1, keepdim=True)  # (b, 1, self.nr_frames, coord_feature_dim)
        # message passed should substract itself, and normalize to it as a single feature

        spatial_message = (spatial_message - bf) / (self.nr_boxes - 1)  # message passed should substract itself
        bf_and_message = torch.cat([bf, spatial_message], dim=3)  # (b, nr_boxes, nr_frames, 2*coord_feature_dim)

        # (b*nr_boxes*nr_frames, coord_feature_dim)
        bf_spatial = self.spatial_node_fusion(bf_and_message.view(b * self.nr_boxes * self.nr_frames, -1))
        bf_spatial = bf_spatial.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)

        bf_temporal_input = bf_spatial.view(b, self.nr_boxes, self.nr_frames * self.coord_feature_dim)

        bf_nonlocal = self.box_feature_fusion(
            bf_temporal_input.view(b * self.nr_boxes, -1))  # (b*nr_boxes, coord_feature_dim)
        bf_nonlocal = bf_nonlocal.view(b, self.nr_boxes, self.coord_feature_dim).permute(0, 2,
                                                                                         1).contiguous()  # (N, C, NB)
        for i in range(self.nr_nonlocal_layers):
            bf_nonlocal = self.nonlocal_fusion[i](bf_nonlocal)

        box_features = torch.mean(bf_nonlocal, dim=2)  # (b, coord_feature_dim)

        # video_features = torch.cat([global_features, local_features, box_features], dim=1)
        video_features = box_features

        cls_output = self.classifier(video_features)  # (b, num_classes)
        return cls_output

class VideoModelGlobalCoordLatent(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, opt,
                 ):
        super(VideoModelGlobalCoordLatent, self).__init__()

        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames
        self.img_feature_dim = opt.img_feature_dim
        self.coord_feature_dim = opt.coord_feature_dim
        self.i3D = Net(self.nr_actions, extract_features=True, loss_type='softmax')
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv3d(2048, 512, kernel_size=(1, 1, 1), stride=1)

        self.category_embed_layer = nn.Embedding(3, opt.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)

        self.c_coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim // 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.c_spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim * 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.c_box_feature_fusion = nn.Sequential(
            nn.Linear((self.nr_frames // 2) * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim + 2*self.img_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )
        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)
        if opt.restore_i3d:
            self.restore_i3d(opt.restore_i3d)
        if opt.restore_custom:
            self.restore_custom(opt.restore_custom)

    def train(self, mode=True):  # overriding default train function
        super(VideoModelGlobalCoordLatent, self).train(mode)
        for m in self.modules():  # or self.modules(), if freezing all bn layers
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def restore_custom(self, restore_path):
        print("restoring path {}".format(restore_path))
        weights = torch.load(restore_path)

        ks = list(weights.keys())
        print('\n\n BEFORE', weights[ks[0]][0,0,0])
        new_weights = {}
        # import pdb
        for k, v in weights.items():
            new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('\n\n AFTER', self.state_dict()[ks[0]][0,0, 0])
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not name.startswith('classifier') :
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'


    def restore_i3d(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        # import pdb
        for k, v in weights.items():
            if 'i3D' in k :
                new_weights[k.replace('module.', '')] = v
        # pdb.set_trace()
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        for m in self.i3D.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        frozen_weights = 0
        for name, param in self.named_parameters():
            if 'i3D' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        # import pdb
        for k, v in weights.items():
            if not 'classifier.4' in k and 'i3D.classifier':
                new_weights[k.replace('module.', '')] = v
        # pdb.set_trace()
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, box_categories, box_input, video_label, is_inference=False):

        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """

        # org_features - [V x 2048 x T / 2 x 14 x 14]
        bs, _, _, _, _ = global_img_input.shape
        y_i3d, org_features = self.i3D(global_img_input)
        # Reduce dimension video_features - [V x 512 x T / 2 x 14 x 14]
        videos_features = self.conv(org_features)
        b = bs

        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b * self.nr_boxes * (self.nr_frames//2), 4)

        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(b * self.nr_boxes * (self.nr_frames // 2))
        box_category_embeddings = self.category_embed_layer(box_categories)  # (b*nr_b*nr_f, coord_feature_dim//2)

        bf = self.c_coord_to_feature(box_input)
        bf = torch.cat([bf, box_category_embeddings], dim=1)  # (b*nr_b*nr_f, coord_feature_dim + coord_feature_dim//2)
        bf = self.c_coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)

        bf = bf.view(b, self.nr_boxes, self.nr_frames // 2, self.coord_feature_dim)

        # spatial message passing (graph)
        spatial_message = bf.sum(dim=1, keepdim=True)  # (b, 1, self.nr_frames, coord_feature_dim)
        # message passed should substract itself, and normalize to it as a single feature
        spatial_message = (spatial_message - bf) / (self.nr_boxes - 1)  # message passed should substract itself

        bf_message_gf = torch.cat([bf, spatial_message], dim=3)  # (b, nr_boxes, nr_frames, 2*coord_feature_dim)

        # (b*nr_boxes*nr_frames, coord_feature_dim)
        bf_spatial = self.c_spatial_node_fusion(bf_message_gf.view(b * self.nr_boxes * (self.nr_frames // 2), -1))
        bf_spatial = bf_spatial.view(b, self.nr_boxes, self.nr_frames // 2, self.coord_feature_dim)

        bf_temporal_input = bf_spatial.view(b, self.nr_boxes, (self.nr_frames // 2) * self.coord_feature_dim)

        box_features = self.c_box_feature_fusion(
            bf_temporal_input.view(b * self.nr_boxes, -1))  # (b*nr_boxes, img_feature_dim)
        coord_ft = torch.mean(box_features.view(b, self.nr_boxes, -1), dim=1)  # (b, coord_feature_dim)
        # video_features = torch.cat([global_features, local_features, box_features], dim=1)
        # _gf = self.global_new_fc(_gf)
        _gf = videos_features.mean(-1).mean(-1).view(b, (self.nr_frames//2), 2*self.img_feature_dim)
        _gf = _gf.mean(1)
        video_features = torch.cat([_gf.view(b, -1), coord_ft], dim=-1)

        cls_output = self.classifier(video_features)  # (b, num_classes)
        return cls_output

class VideoModelGlobalCoordLatentNL(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, base_net, opt,
                 ):
        super(VideoModelGlobalCoordLatentNL, self).__init__()

        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames
        self.img_feature_dim = opt.img_feature_dim
        self.coord_feature_dim = opt.coord_feature_dim
        self.i3D = Net(self.nr_actions, extract_features=True, loss_type='softmax')
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv3d(2048, 512, kernel_size=(1, 1, 1), stride=1)


        self.category_embed_layer = nn.Embedding(3, opt.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)

        self.c_coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim // 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.c_spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim * 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.nr_nonlocal_layers = 3
        self.c_nonlocal_fusion = []
        for i in range(self.nr_nonlocal_layers):
            self.c_nonlocal_fusion.append(nn.Sequential(
                    Nonlocal(dim=self.coord_feature_dim, dim_inner=self.coord_feature_dim // 2),
                    nn.Conv1d(self.coord_feature_dim, self.coord_feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm1d(self.coord_feature_dim),
                    nn.ReLU()
            ))
        self.c_nonlocal_fusion = nn.ModuleList(self.c_nonlocal_fusion)

        self.c_box_feature_fusion = nn.Sequential(
            nn.Linear((self.nr_frames // 2) * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim + 2*self.img_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )
        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)
        if opt.restore_i3d:
            self.restore_i3d(opt.restore_i3d)

        if opt.restore_custom:
            self.restore_custom(opt.restore_custom)

    def restore_custom(self, restore_path):
        print("restoring path {}".format(restore_path))
        weights = torch.load(restore_path)
        ks = list(weights.keys())
        print('\n\n BEFORE', weights[ks[0]][0,0,0])
        new_weights = {}
        # import pdb
        for k, v in weights.items():
            new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('\n\n AFTER', self.state_dict()[ks[0]][0,0, 0])
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not name.startswith('classifier') :
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'



    def restore_i3d(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        # import pdb
        for k, v in weights.items():
            if 'i3D' in k  or k.startswith('conv.'):
                new_weights[k.replace('module.', '')] = v
        # pdb.set_trace()
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        for m in self.i3D.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        frozen_weights = 0
        for name, param in self.named_parameters():
            if 'i3D' in name or k.startswith('conv.') :
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def train(self, mode=True):  # overriding default train function
        super(VideoModelGlobalCoordLatentNL, self).train(mode)
        for m in self.i3D.modules():  # or self.modules(), if freezing all bn layers
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        import pdb
        for k, v in weights.items():
            if not 'classifier.4' in k and 'i3D.classifier' not in k:
                new_weights[k.replace('module.', '')] = v
        pdb.set_trace()
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, box_categories, box_input, video_label, is_inference=False):

        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """

        # org_features - [V x 2048 x T / 2 x 14 x 14]
        bs, _, _, _, _ = global_img_input.shape
        y_i3d, org_features = self.i3D(global_img_input)
        # Reduce dimension video_features - [V x 512 x T / 2 x 14 x 14]
        videos_features = self.conv(org_features)
        b = bs

        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b * self.nr_boxes * (self.nr_frames//2), 4)

        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(b * self.nr_boxes * (self.nr_frames // 2))
        box_category_embeddings = self.category_embed_layer(box_categories)  # (b*nr_b*nr_f, coord_feature_dim//2)

        bf = self.c_coord_to_feature(box_input)
        bf = torch.cat([bf, box_category_embeddings], dim=1)  # (b*nr_b*nr_f, coord_feature_dim + coord_feature_dim//2)
        bf = self.c_coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)

        bf = bf.view(b, self.nr_boxes, self.nr_frames // 2, self.coord_feature_dim)

        # spatial message passing (graph)
        spatial_message = bf.sum(dim=1, keepdim=True)  # (b, 1, self.nr_frames, coord_feature_dim)
        # message passed should substract itself, and normalize to it as a single feature
        spatial_message = (spatial_message - bf) / (self.nr_boxes - 1)  # message passed should substract itself

        bf_message_gf = torch.cat([bf, spatial_message], dim=3)  # (b, nr_boxes, nr_frames, 2*coord_feature_dim)

        # (b*nr_boxes*nr_frames, coord_feature_dim)
        bf_spatial = self.c_spatial_node_fusion(bf_message_gf.view(b * self.nr_boxes * (self.nr_frames // 2), -1))
        bf_spatial = bf_spatial.view(b, self.nr_boxes, self.nr_frames // 2, self.coord_feature_dim)

        bf_temporal_input = bf_spatial.view(b, self.nr_boxes, (self.nr_frames // 2) * self.coord_feature_dim)

        bf_nonlocal = self.c_box_feature_fusion(
            bf_temporal_input.view(b * self.nr_boxes, -1))  # (b*nr_boxes, img_feature_dim)

        bf_nonlocal = bf_nonlocal.view(b, self.nr_boxes, self.coord_feature_dim).permute(0, 2, 1).contiguous()  # (N, C, NB)
        for i in range(self.nr_nonlocal_layers):
            bf_nonlocal = self.c_nonlocal_fusion[i](bf_nonlocal)

        coord_ft = torch.mean(bf_nonlocal, dim=2)  # (b, coord_feature_dim)

        # video_features = torch.cat([global_features, local_features, box_features], dim=1)
        _gf = videos_features.mean(-1).mean(-1).view(b, (self.nr_frames//2), 2*self.img_feature_dim)
        _gf = _gf.mean(1)
        video_features = torch.cat([_gf.view(b, -1), coord_ft], dim=-1)

        cls_output = self.classifier(video_features)  # (b, num_classes)
        return cls_output

class VideoGlobalModel(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, opt,
                 ):
        super(VideoGlobalModel, self).__init__()

        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames
        self.img_feature_dim = opt.img_feature_dim
        self.coord_feature_dim = opt.coord_feature_dim
        self.i3D = Net(self.nr_actions, extract_features=True, loss_type='softmax')
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv3d(2048, 512, kernel_size=(1, 1, 1), stride=1)
        self.fc = nn.Linear(512, self.nr_actions)
        self.crit = nn.CrossEntropyLoss()

        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, local_img_input, box_input, video_label, is_inference=False):
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """

        # org_features - [V x 2048 x T / 2 x 14 x 14]
        y_i3d, org_features = self.i3D(global_img_input)
        # Reduce dimension video_features - [V x 512 x T / 2 x 14 x 14]
        videos_features = self.conv(org_features)

        # Get global features - [V x 512]
        global_features = self.avgpool(videos_features).squeeze()
        global_features = self.dropout(global_features)

        cls_output = self.fc(global_features)
        return cls_output

class VideoModelGlobalCoord(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, opt):
        super(VideoModelGlobalCoord, self).__init__()

        self.nr_boxes = opt.num_boxes
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames
        self.img_feature_dim = opt.img_feature_dim
        self.coord_feature_dim = opt.coord_feature_dim
        self.i3D = Net(self.nr_actions, extract_features=True, loss_type='softmax')
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv3d(2048, 256, kernel_size=(1, 1, 1), stride=1)


        self.global_new_fc = nn.Sequential(
            nn.Linear(256, self.img_feature_dim, bias=False),
            nn.BatchNorm1d(self.img_feature_dim),
            nn.ReLU(inplace=True)
        )


        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim // 2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim // 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.c_spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim * 2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.c_box_feature_fusion = nn.Sequential(
            nn.Linear((self.nr_frames // 2) * self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim + self.img_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )
        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)
        if opt.restore_i3d:
            self.restore_i3d(opt.restore_i3d)

    def train(self, mode=True):  # overriding default train function
        super(VideoModelGlobalCoord, self).train(mode)
        for m in self.i3D.modules():  # or self.modules(), if freezing all bn layers
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.eval()
                # shutdown update in frozen mode
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def restore_i3d(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        # import pdb
        for k, v in weights.items():
            if 'i3D' in k :
                new_weights[k.replace('module.', '')] = v
        # pdb.set_trace()
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if 'i3D' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        # import pdb
        for k, v in weights.items():
            if not 'classifier.4' in k and 'i3D.classifier':
                new_weights[k.replace('module.', '')] = v
        # pdb.set_trace()
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def forward(self, global_img_input, box_categories, box_input, video_label, is_inference=False):

        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """

        # org_features - [V x 2048 x T / 2 x 14 x 14]
        bs, _, _, _, _ = global_img_input.shape
        y_i3d, org_features = self.i3D(global_img_input)
        # Reduce dimension video_features - [V x 512 x T / 2 x 14 x 14]
        videos_features = self.conv(org_features)
        b = bs

        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b * self.nr_boxes * (self.nr_frames//2), 4)

        bf = self.c_coord_to_feature(box_input)
        bf = bf.view(b, self.nr_boxes, self.nr_frames // 2, self.coord_feature_dim)

        # spatial message passing (graph)
        spatial_message = bf.sum(dim=1, keepdim=True)  # (b, 1, self.nr_frames, coord_feature_dim)
        # message passed should substract itself, and normalize to it as a single feature
        spatial_message = (spatial_message - bf) / (self.nr_boxes - 1)  # message passed should substract itself

        bf_message_gf = torch.cat([bf, spatial_message], dim=3)  # (b, nr_boxes, nr_frames, 2*coord_feature_dim)

        # (b*nr_boxes*nr_frames, coord_feature_dim)
        bf_spatial = self.c_spatial_node_fusion(bf_message_gf.view(b * self.nr_boxes * (self.nr_frames // 2), -1))
        bf_spatial = bf_spatial.view(b, self.nr_boxes, self.nr_frames // 2, self.coord_feature_dim)

        bf_temporal_input = bf_spatial.view(b, self.nr_boxes, (self.nr_frames // 2) * self.coord_feature_dim)

        box_features = self.c_box_feature_fusion(
            bf_temporal_input.view(b * self.nr_boxes, -1))  # (b*nr_boxes, img_feature_dim)
        coord_ft = torch.mean(box_features.view(b, self.nr_boxes, -1), dim=1)  # (b, coord_feature_dim)
        # video_features = torch.cat([global_features, local_features, box_features], dim=1)
        _gf = videos_features.mean(-1).mean(-1).view(b*(self.nr_frames//2), self.img_feature_dim)
        _gf = self.global_new_fc(_gf)
        _gf = _gf.view(b, self.nr_frames // 2, self.img_feature_dim).mean(1)
        video_features = torch.cat([_gf.view(b, -1), coord_ft], dim=-1)

        cls_output = self.classifier(video_features)  # (b, num_classes)
        return cls_output
