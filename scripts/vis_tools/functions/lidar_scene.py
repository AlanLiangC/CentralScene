import os
import json
import torch
from omegaconf import OmegaConf
from model.SGDiff import SGDiff
from helpers.util import postprocess_sincos2arctan

class LidarScene:
    def __init__(self, exp_path, epoch, dataset):
        self.exp_path = exp_path
        self.epoch = epoch
        self.dataset = dataset
        argsJson = os.path.join(exp_path, 'args.json')
        assert os.path.exists(argsJson), 'Could not find args.json for experiment {}'.format(exp_path)
        with open(argsJson) as j:
            self.modelArgs = json.load(j)

    def build_model(self):
        # instantiate the model
        diff_opt = self.modelArgs['diff_yaml']
        diff_cfg = OmegaConf.load(diff_opt)
        with_changes_ = self.modelArgs['with_changes'] if 'with_changes' in self.modelArgs else None
        modeltype_ = self.modelArgs['network_type']
        replacelatent_ = self.modelArgs['replace_latent'] if 'replace_latent' in self.modelArgs else None

        # diff_cfg.layout_branch.diffusion_kwargs.train_stats_file = test_dataset_no_changes.box_normalized_stats
        diff_cfg.layout_branch.denoiser_kwargs.using_clip = self.modelArgs['with_CLIP']
        self.model = SGDiff(type=modeltype_, diff_opt=diff_cfg, vocab=self.dataset.vocab, replace_latent=replacelatent_,
                    with_changes=with_changes_, residual=self.modelArgs['residual'], gconv_pooling=self.modelArgs['pooling'], clip=self.modelArgs['with_CLIP'],
                    with_angles=self.modelArgs['with_angles'], separated=self.modelArgs['separated'])
        self.model.diff.optimizer_ini()
        self.model.load_networks(exp=self.exp_path, epoch=self.epoch, restart_optim=True, load_shape_branch=False)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model = self.model.eval()

    def inference_sample(self, sample):
        dec_objs, dec_triples = sample['decoder']['objs'], sample['decoder']['tripltes']
        dec_objs, dec_triples = dec_objs.cuda(), dec_triples.cuda()
        encoded_dec_text_feat, encoded_dec_rel_feat = None, None
        encoded_dec_text_feat, encoded_dec_rel_feat = sample['decoder']['text_feats'].cuda(), sample['decoder']['rel_feats'].cuda()
        with torch.no_grad():

            data_dict = self.model.sample_box_and_shape(dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, gen_shape=False)

            boxes_pred, angles_pred = torch.concat((data_dict['sizes'],data_dict['translations']),dim=-1), data_dict['angles']
            angles_pred = postprocess_sincos2arctan(angles_pred)
            boxes_pred_den = self.dataset.re_scale_box(torch.concat([boxes_pred, angles_pred], dim=-1))
            return boxes_pred_den