import datetime
import os
import random
import time

import numpy as np
import torch.amp
from tqdm import tqdm

import torch
import torch.distributed
from tools.data import build_dataloader
from tools.utils.ckpt import load_ckpt, save_ckpt
from tools.utils.logging import get_logger
from tools.utils.stats import TrainingStats
from tools.utils.utility import AverageMeter

__all__ = ['Trainer']

import torch.distributed as dist

rank = int(os.environ.get('RANK', 0))  # torchrun 会提供 RANK


def is_main_process():
    return (not dist.is_available() or not dist.is_initialized() or rank == 0)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class Trainer(object):

    def __init__(self, cfg, mode='train', task='rec'):
        self.cfg = cfg.cfg
        self.task = task
        self.local_rank = (int(os.environ['LOCAL_RANK'])
                           if 'LOCAL_RANK' in os.environ else 0)
        self.set_device(self.cfg['Global']['device'])
        mode = mode.lower()
        assert mode in [
            'train_eval',
            'train',
            'eval',
            'test',
        ], 'mode should be train, eval and test'
        if torch.cuda.device_count() > 1 and 'train' in mode:
            torch.distributed.init_process_group(backend='nccl')
            torch.cuda.set_device(self.device)
            self.cfg['Global']['distributed'] = True
        else:
            self.cfg['Global']['distributed'] = False
            self.local_rank = 0

        # Auto-split train batch across GPUs if Global.total_bs is set.
        # (yml의 first_bs/batch_size_per_card를 world_size로 나눠 per-card bs 산출)
        total_bs = self.cfg['Global'].get('total_bs')
        if total_bs and 'train' in mode:
            world_size = (torch.distributed.get_world_size()
                          if self.cfg['Global']['distributed'] else 1)
            per_card = int(total_bs) // world_size
            assert per_card > 0, (
                f'total_bs={total_bs} too small for world_size={world_size}')
            if int(total_bs) % world_size != 0:
                print(f'[WARN] total_bs={total_bs} not divisible by '
                      f'world_size={world_size}; using per_card={per_card} '
                      f'(effective total={per_card * world_size})')
            train_loader = self.cfg.get('Train', {}).get('loader')
            train_sampler = self.cfg.get('Train', {}).get('sampler')
            if train_loader is not None:
                train_loader['batch_size_per_card'] = per_card
            if train_sampler is not None:
                train_sampler['first_bs'] = per_card

        self.cfg['Global']['output_dir'] = self.cfg['Global'].get(
            'output_dir', 'output')
        tag = self.cfg['Global'].get('tag', '')
        if tag:
            # Append tag to output_dir basename, preserving trailing slash
            out_dir = self.cfg['Global']['output_dir'].rstrip('/')
            self.cfg['Global']['output_dir'] = out_dir + tag
        # Only training writes into output_dir (config.yml, train.log, tb, ckpts).
        # Pure eval shouldn't pollute it with empty folders.
        if 'train' in mode:
            os.makedirs(self.cfg['Global']['output_dir'], exist_ok=True)

        self.writer = None
        self.wandb_run = None
        if is_main_process() and 'train' in mode:
            if self.cfg['Global']['use_tensorboard']:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.cfg['Global']['output_dir'])

        self.logger = get_logger(
            'openrec' if task == 'rec' else 'opendet',
            os.path.join(self.cfg['Global']['output_dir'], 'train.log')
            if 'train' in mode else None,
        )

        cfg.print_cfg(self.logger.info)

        if self.cfg['Global']['device'] == 'gpu' and self.device.type == 'cpu':
            self.logger.info('cuda is not available, auto switch to cpu')

        self.set_random_seed(self.cfg['Global'].get('seed', 48))

        # build data loader
        self.train_dataloader = None
        if 'train' in mode:
            if is_main_process():
                cfg.save(
                    os.path.join(self.cfg['Global']['output_dir'],
                                 'config.yml'), self.cfg)
            self.train_dataloader = build_dataloader(self.cfg,
                                                     'Train',
                                                     self.logger,
                                                     task=task)
            self.logger.info(
                f'train dataloader has {len(self.train_dataloader)} iters')
        self.valid_dataloader = None
        if 'eval' in mode and self.cfg['Eval']:
            try:
                self.valid_dataloader = build_dataloader(self.cfg,
                                                        'Eval',
                                                        self.logger,
                                                        task=task)
                self.logger.info(
                    f'valid dataloader has {len(self.valid_dataloader)} iters')
            except Exception as e:
                self.logger.info(f'valid dataloader build failed: {e}')
                self.valid_dataloader = None

        self.pl_charset_adapter = None
        if task == 'rec':
            self._init_rec_model()
        elif task == 'det':
            self._init_det_model()
        else:
            raise NotImplementedError

        self.logger.info(get_parameter_number(model=self.model))
        self.model = self.model.to(self.device)

        use_sync_bn = self.cfg['Global'].get('use_sync_bn', False)
        if use_sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)
            self.logger.info('convert_sync_batchnorm')
        self.accumulation_steps = self.cfg['Global'].get(
            'accumulation_steps', 1)
        from openrec.optimizer import build_optimizer
        self.optimizer, self.lr_scheduler = None, None
        epochs = self.cfg['Global']['epoch_num']
        try:
            step_each_epoch = len(self.train_dataloader)
        except TypeError:
            # 针对 IterableDataset 的处理
            step_each_epoch = self.cfg['Global'].get('total_iter_steps', 100000)
        if self.train_dataloader is not None:
            # build optim
            self.optimizer, self.lr_scheduler = build_optimizer(
                self.cfg['Optimizer'],
                self.cfg['LRScheduler'],
                epochs=epochs,
                step_each_epoch=step_each_epoch,
                model=self.model,
            )
        self.grad_clip_val = self.cfg['Global'].get('grad_clip_val', 0)

        self.status = load_ckpt(self.model, self.cfg, self.optimizer,
                                self.lr_scheduler, mode=mode)
        if (is_main_process() and 'train' in mode
                and self.cfg['Global'].get('use_wandb', False)):
            self._init_wandb()

        # torch.compile (applied AFTER load_ckpt so ckpt loads on raw keys, and
        # BEFORE DDP so self.model.module remains the OptimizedModule — keeps
        # the existing eval-time `self.model.module(...)` bypass intact).
        use_compile = self.cfg['Global'].get('use_compile', False)
        if use_compile and 'train' in mode:
            compile_mode = self.cfg['Global'].get('compile_mode', 'default')
            compile_dynamic = self.cfg['Global'].get('compile_dynamic', False)
            self.model = torch.compile(self.model,
                                       mode=compile_mode,
                                       dynamic=compile_dynamic)
            self.logger.info(
                f'torch.compile enabled '
                f'(mode={compile_mode}, dynamic={compile_dynamic})')

        if self.cfg['Global']['distributed']:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, [self.local_rank], find_unused_parameters=False)

        # amp
        self.scaler = (torch.cuda.amp.GradScaler() if self.cfg['Global'].get(
            'use_amp', False) else None)

        self.logger.info(
            f'run with torch {torch.__version__} and device {self.device}')

    def _init_wandb(self):
        import wandb

        global_cfg = self.cfg['Global']
        run_name = global_cfg.get(
            'run_name',
            os.path.basename(global_cfg['output_dir'].rstrip('/')),
        )
        wandb_config = dict(self.cfg)
        wandb_config['warmup_epoch'] = self.cfg.get('LRScheduler', {}).get(
            'warmup_epoch')
        wandb_config['epoch_num'] = global_cfg.get('epoch_num')
        wandb_config['lr'] = self.cfg.get('Optimizer', {}).get('lr')
        wandb_config['world_size'] = (torch.distributed.get_world_size()
                                      if global_cfg.get('distributed') else 1)

        run_id_file = global_cfg.get('wandb_run_id_file')
        if not run_id_file:
            run_id_file = os.path.join(global_cfg['output_dir'],
                                       'wandb_run_id.txt')
        run_id_file = os.path.expanduser(run_id_file)

        explicit_run_id = global_cfg.get('wandb_run_id') or os.environ.get(
            'WANDB_RUN_ID')
        explicit_run_id = explicit_run_id.strip() if explicit_run_id else None

        file_run_id = None
        if os.path.exists(run_id_file):
            with open(run_id_file, 'r') as f:
                file_run_id = f.read().strip() or None

        should_resume = bool(global_cfg.get('wandb_resume', False))
        run_id = explicit_run_id or (file_run_id if should_resume else None)

        init_kwargs = {
            'project': global_cfg.get('wandb_project', 'ELC'),
            'name': run_name,
            'config': wandb_config,
            'dir': global_cfg.get('wandb_dir', '/tmp'),
        }
        if global_cfg.get('wandb_entity'):
            init_kwargs['entity'] = global_cfg.get('wandb_entity')
        if global_cfg.get('wandb_group'):
            init_kwargs['group'] = global_cfg.get('wandb_group')
        if global_cfg.get('wandb_tags'):
            init_kwargs['tags'] = global_cfg.get('wandb_tags')

        use_rewind = False
        if should_resume and run_id:
            resume_step = global_cfg.get('wandb_resume_step',
                                         self.status.get('global_step'))
            if global_cfg.get('wandb_rewind', False) and resume_step is not None:
                init_kwargs['resume_from'] = f'{run_id}?_step={int(resume_step)}'
                init_kwargs['allow_val_change'] = True
                use_rewind = True
                self.logger.info(
                    f'wandb rewind/resume: run_id={run_id}, step={int(resume_step)}')
            else:
                init_kwargs['id'] = run_id
                init_kwargs['resume'] = global_cfg.get('wandb_resume_mode',
                                                       'allow')
                init_kwargs['allow_val_change'] = True
                self.logger.info(
                    f'wandb resume: run_id={run_id}, mode={init_kwargs["resume"]}')
        elif should_resume:
            self.logger.info(
                f'wandb resume requested, but no run id found at {run_id_file}; '
                'starting a new wandb run')
        elif explicit_run_id:
            init_kwargs['id'] = explicit_run_id

        try:
            self.wandb_run = wandb.init(**init_kwargs)
        except wandb.errors.CommError as e:
            if use_rewind and 'rewind' in str(e).lower():
                self.logger.info(
                    f'wandb rewind unavailable ({e}); '
                    'falling back to plain resume')
                init_kwargs.pop('resume_from', None)
                init_kwargs['id'] = run_id
                init_kwargs['resume'] = global_cfg.get('wandb_resume_mode',
                                                       'allow')
                # Tear down partially-initialized wandb state from the failed
                # rewind attempt; otherwise the retry inherits broken globals.
                try:
                    wandb.teardown()
                except Exception:
                    pass
                self.wandb_run = wandb.init(**init_kwargs)
            else:
                raise

        try:
            os.makedirs(os.path.dirname(run_id_file), exist_ok=True)
            with open(run_id_file, 'w') as f:
                f.write(f'{self.wandb_run.id}\n')
        except OSError as e:
            self.logger.info(
                f'failed to write wandb run id file {run_id_file}: {e}')

        # NOTE: we deliberately do NOT call `define_metric('*', step_metric=
        # 'global_step')`. Doing so retroactively breaks pre-existing runs:
        # historical points have no `global_step` field, so they vanish from
        # charts when the x-axis is forced to global_step. Instead, log calls
        # below omit the explicit `step=` arg (so wandb's internal `_step`
        # keeps monotonically incrementing past any prior end-step) and
        # include `global_step` as a regular metric. Default charts stay on
        # the `_step` axis (old data preserved); switch the x-axis to
        # `global_step` in the wandb UI to view new-run data aligned to
        # training step.

    def _init_rec_model(self):
        from openrec.losses import build_loss as build_rec_loss
        from openrec.metrics import build_metric as build_rec_metric
        from openrec.modeling import build_model as build_rec_model
        from openrec.postprocess import build_post_process as build_rec_post_process

        # CCD mode: extend charset with Unicode variants before building post_process
        unicode_mapping_path = self.cfg['Global'].get('unicode_mapping', None)
        if unicode_mapping_path:
            from tools.utils.charset_utils import build_pl_charset, PLCharsetAdapter, generate_extended_dict
            base_dict_path = self.cfg['Global']['character_dict_path']
            ext_dict_path = generate_extended_dict(base_dict_path, unicode_mapping_path)
            self.cfg['Global']['character_dict_path'] = ext_dict_path
            # Also update any LabelEncode transforms that reference the dict
            for section in ('Train', 'Eval'):
                if section in self.cfg and 'dataset' in self.cfg[section]:
                    transforms = self.cfg[section]['dataset'].get('transforms', [])
                    for t in transforms:
                        if isinstance(t, dict):
                            for name, params in t.items():
                                if 'Encode' in name and isinstance(params, dict) and 'character_dict_path' in params:
                                    params['character_dict_path'] = ext_dict_path
            self.logger.info(f'CCD mode: extended dict at {ext_dict_path}')

        # build post process
        self.post_process_class = build_rec_post_process(
            self.cfg['PostProcess'], self.cfg['Global'])

        # CCD mode: build PLCharsetAdapter for eval
        self.pl_charset_adapter = None
        if unicode_mapping_path:
            _, ext_to_base = build_pl_charset(base_dict_path, unicode_mapping_path)
            # Target charset = base charset (from original dict)
            with open(base_dict_path, 'r', encoding='utf-8') as f:
                base_chars = [line.strip() for line in f.readlines()]
            self.pl_charset_adapter = PLCharsetAdapter(''.join(base_chars), ext_to_base)
            self.logger.info(f'CCD mode: PLCharsetAdapter with {len(ext_to_base)} extended chars')

        # build model
        # for rec algorithm
        self.use_transformers = self.cfg['Global'].get('use_transformers',
                                                       False)
        if self.use_transformers:
            if self.cfg['Architecture']['algorithm'] == 'UniRec':
                from openrec.modeling.unirec_modeling.modeling_unirec import UniRecForConditionalGenerationNew
                from openrec.modeling.unirec_modeling.configuration_unirec import UniRecConfig
                cfg_vlm = UniRecConfig.from_pretrained(
                    self.cfg['Global']['vlm_ocr_config'])
                cfg_vlm._attn_implementation = 'flash_attention_2'
                # cfg_vlm._attn_implementation = "eager"
                # cfg_vlm._attn_implementation = "sdpa"
                self.model = UniRecForConditionalGenerationNew(config=cfg_vlm)
            elif self.cfg['Architecture']['algorithm'] == 'CMER':
                from openrec.modeling.cmer_modeling.modeling_cmer import CMER, CMERConfig
                cfg_model = CMERConfig(
                    self.cfg['Architecture']['vision_config'],
                    self.cfg['Architecture']['decoder_config'])
                self.model = CMER(config=cfg_model)
        else:
            char_num = self.post_process_class.get_character_num()
            self.cfg['Architecture']['Decoder']['out_channels'] = char_num
            self.model = build_rec_model(self.cfg['Architecture'])
        # build loss
        self.loss_class = build_rec_loss(self.cfg['Loss'])
        # build metric
        self.eval_class = build_rec_metric(self.cfg['Metric'])

    def _init_det_model(self):
        from opendet.losses import build_loss as build_det_loss
        from opendet.metrics import build_metric as build_det_metric
        from opendet.modeling import build_model as build_det_model
        from opendet.postprocess import build_post_process as build_det_post_process

        # build post process
        self.post_process_class = build_det_post_process(
            self.cfg['PostProcess'], self.cfg['Global'])
        # build detmodel
        self.model = build_det_model(self.cfg['Architecture'])
        # build loss
        self.loss_class = build_det_loss(self.cfg['Loss'])
        # build metric
        self.eval_class = build_det_metric(self.cfg['Metric'])

    def load_params(self, params):
        self.model.load_state_dict(params)

    def set_random_seed(self, seed):
        torch.manual_seed(seed)  # 为CPU设置随机种子
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        random.seed(seed)
        np.random.seed(seed)

    def set_device(self, device):
        if device == 'gpu' and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.local_rank}')
        else:
            device = torch.device('cpu')
        self.device = device

    def train(self):
        cal_metric_during_train = self.cfg['Global'].get(
            'cal_metric_during_train', False)
        log_smooth_window = self.cfg['Global']['log_smooth_window']
        epoch_num = self.cfg['Global']['epoch_num']
        print_batch_step = self.cfg['Global']['print_batch_step']
        eval_epoch_step = self.cfg['Global'].get('eval_epoch_step', 1)

        start_eval_epoch = 0
        if self.valid_dataloader is not None:
            if type(eval_epoch_step) == list and len(eval_epoch_step) >= 2:
                start_eval_epoch = eval_epoch_step[0]
                eval_epoch_step = eval_epoch_step[1]
                if len(self.valid_dataloader) == 0:
                    start_eval_epoch = 1e111
                    self.logger.info(
                        'No Images in eval dataset, evaluation during training will be disabled'
                    )
                self.logger.info(
                    f'During the training process, after the {start_eval_epoch}th epoch, '
                    f'an evaluation is run every {eval_epoch_step} epoch')
        else:
            start_eval_epoch = 1e111

        eval_batch_step = self.cfg['Global']['eval_batch_step']

        global_step = self.status.get('global_step', 0)

        start_eval_step = 0
        if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
            start_eval_step = eval_batch_step[0]
            eval_batch_step = eval_batch_step[1]
            if self.valid_dataloader is not None and len(self.valid_dataloader) == 0:
                self.logger.info(
                    'No Images in eval dataset, evaluation during training '
                    'will be disabled')
                start_eval_step = 1e111
            self.logger.info(
                'During the training process, after the {}th iteration, '
                'an evaluation is run every {} iterations'.format(
                    start_eval_step, eval_batch_step))

        save_epoch_step = self.cfg['Global'].get('save_epoch_step', [0, 1])
        start_save_epoch = save_epoch_step[0]
        save_epoch_step = save_epoch_step[1]

        start_epoch = self.status.get('epoch', 1)
        self.best_metric = self.status.get('metrics', {})
        if self.eval_class.main_indicator not in self.best_metric:
            self.best_metric[self.eval_class.main_indicator] = 0
        train_stats = TrainingStats(log_smooth_window, ['lr'])
        self.model.train()

        total_samples = 0
        train_reader_cost = 0.0
        train_batch_cost = 0.0
        reader_start = time.time()
        eta_meter = AverageMeter()
        save_iter_step = self.cfg['Global'].get('save_iter_step',
                                                [10e10, 2000])
        start_save_iter = save_iter_step[0]
        save_iter_step = save_iter_step[1]

        if self.cfg['Global'].get('resume_from_iter',
                                  False):  # for unirec resume training
            if self.cfg['Global']['checkpoints'] is None:
                raise ValueError(
                    'resume_from_iter is True, but checkpoints is None')
            start_epoch = start_epoch - 1
            self.resume_iter = global_step
            iter_model_file_name = os.path.basename(
                self.cfg['Global']['checkpoints'])
            last_whole_epoch_global_step = iter_model_file_name.split('_')[1]
            self.cfg['Train']['sampler'][
                'resume_iter'] = self.resume_iter - last_whole_epoch_global_step

        fresh_param_prefixes = self.cfg['Global'].get('fresh_params', [])
        freeze_param_prefixes = self.cfg['Global'].get('freeze_params', [])
        if fresh_param_prefixes and freeze_param_prefixes:
            overlap = set(fresh_param_prefixes) & set(freeze_param_prefixes)
            if overlap:
                raise ValueError(
                    f'fresh_params and freeze_params overlap: {overlap}')
        freeze_epochs = self.cfg['Global'].get('freeze_epochs', None)
        try:
            _step_each_epoch = len(self.train_dataloader)
        except TypeError:
            _step_each_epoch = self.cfg['Global'].get('total_iter_steps', 100000)
        freeze_steps = int(freeze_epochs * _step_each_epoch) if freeze_epochs is not None else 0
        backbone_frozen = False

        last_whole_epoch_global_step = 0
        # initial validation before training
        if self.valid_dataloader is not None and is_main_process():
            self.eval_step(global_step, start_epoch, save_best=False,
                           log_samples=True)

        for epoch in range(start_epoch, epoch_num + 1):

            if not self.cfg['Global'].get('resume_from_iter',
                                          False):  # for unirec resume training
                if 'sampler' in self.cfg['Train']:
                    self.cfg['Train']['sampler']['resume_iter'] = 0
            if hasattr(self.train_dataloader, 'dataset') and self.train_dataloader.dataset is not None:
                if self.train_dataloader.dataset.need_reset and epoch > 1:
                    self.train_dataloader = build_dataloader(self.cfg,
                                                            'Train',
                                                            self.logger,
                                                            epoch=epoch,
                                                            task=self.task)

            for idx, batch in enumerate(self.train_dataloader):
                if self.cfg['Global'].get('resume_from_iter',
                                          False):  # for unirec resume training
                    if global_step != self.resume_iter:
                        global_step += 1
                        if is_main_process(
                        ) and global_step % print_batch_step == 0:
                            self.logger.info(
                                f'skip iter {global_step}, resume from iter {self.resume_iter}'
                            )
                        continue
                    else:
                        global_step += 1
                        self.cfg['Global']['resume_from_iter'] = False
                        self.logger.info(
                            f'resume from iter {self.resume_iter}, start training from iter {global_step}'
                        )
                        continue

                # freeze/unfreeze based on global_step
                if freeze_steps > 0 and freeze_param_prefixes:
                    base_model = self.model.module if self.cfg['Global']['distributed'] else self.model
                    if global_step < freeze_steps and not backbone_frozen:
                        for name, param in base_model.named_parameters():
                            is_frozen = any(name.startswith(p) for p in freeze_param_prefixes)
                            param.requires_grad_(not is_frozen)
                        backbone_frozen = True
                        if is_main_process():
                            self.logger.info(
                                f'freeze_warmup: FROZEN for {freeze_epochs} epochs '
                                f'({freeze_steps} steps), freezing {freeze_param_prefixes}')
                    elif global_step >= freeze_steps and backbone_frozen:
                        for param in base_model.parameters():
                            param.requires_grad_(True)
                        backbone_frozen = False
                        if is_main_process():
                            self.logger.info(
                                f'freeze_warmup: step {global_step} — UNFROZEN, training all params')

                batch_tensor = [t.to(self.device) for t in batch]
                batch_numpy = [t.numpy() for t in batch]
                train_reader_cost += time.time() - reader_start
                # use amp
                if self.scaler:
                    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    with torch.amp.autocast(device_type=self.device.type,
                                            dtype=amp_dtype):
                        if self.use_transformers:
                            inputs = {
                                'pixel_values': batch_tensor[0],
                                'input_ids': None,
                                'attention_mask': None,
                                'labels': batch_tensor[1],
                                'length': batch_tensor[2]
                            }
                            preds = self.model(**inputs)
                        else:
                            preds = self.model(batch_tensor[0],
                                               data=batch_tensor[1:])
                        loss = self.loss_class(preds, batch_tensor)
                        loss['loss'] = loss['loss'] / self.accumulation_steps
                    self.scaler.scale(loss['loss']).backward()
                    if (global_step + 1) % self.accumulation_steps == 0:
                        if self.grad_clip_val > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.grad_clip_val)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    preds = self.model(batch_tensor[0], data=batch_tensor[1:])
                    loss = self.loss_class(preds, batch_tensor)
                    avg_loss = loss['loss']
                    avg_loss.backward()
                    if self.grad_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.grad_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if cal_metric_during_train:  # only rec and cls need
                    # MDiff4STR returns loss scalar during training, not logits
                    preds_is_logits = isinstance(preds, torch.Tensor) and preds.dim() >= 2
                    preds_is_numpy = isinstance(preds, np.ndarray) and preds.ndim >= 2
                    if preds_is_logits or preds_is_numpy:
                        post_result = self.post_process_class(preds,
                                                              batch_numpy,
                                                              training=True)
                        if self.pl_charset_adapter is not None and isinstance(post_result, tuple):
                            preds_list, labels_list = post_result
                            preds_list = [(self.pl_charset_adapter(text), conf) for text, conf in preds_list]
                            post_result = (preds_list, labels_list)
                        self.eval_class(post_result, batch_numpy, training=True)
                        metric = self.eval_class.get_metric()
                        train_stats.update(metric)

                train_batch_time = time.time() - reader_start
                train_batch_cost += train_batch_time
                eta_meter.update(train_batch_time)
                global_step += 1
                total_samples += len(batch[0])

                try:
                    self.lr_scheduler.step()
                except Exception as e:
                    self.logger.info(
                        f'lr_scheduler step error, {e}, please check your config'
                    )

                loss['loss'] = loss['loss'] * self.accumulation_steps
                # logger
                stats = {
                    k: float(v)
                    if v.shape == [] else v.detach().cpu().numpy().mean()
                    for k, v in loss.items()
                }
                try:
                    stats['lr'] = self.lr_scheduler.get_last_lr()[0]
                except Exception:
                    stats['lr'] = 0.0
                train_stats.update(stats)

                if self.writer is not None:
                    for k, v in train_stats.get().items():
                        self.writer.add_scalar(f'TRAIN/{k}', v, global_step)

                if self.wandb_run is not None:
                    _payload = {f'train/{k}': v for k, v in train_stats.get().items()}
                    _payload['global_step'] = global_step
                    self.wandb_run.log(_payload)

                if is_main_process() and (
                    (global_step > 0 and global_step % print_batch_step == 0)
                        or (idx >= len(self.train_dataloader) - 1)):
                    logs = train_stats.log()

                    eta_sec = (
                        (epoch_num + 1 - epoch) * len(self.train_dataloader) -
                        idx - 1) * eta_meter.avg
                    eta_sec_format = str(
                        datetime.timedelta(seconds=int(eta_sec)))
                    strs = (
                        f'epoch: [{epoch}/{epoch_num}], global_step: {global_step}, {logs}, '
                        f'avg_reader_cost: {train_reader_cost / print_batch_step:.5f} s, '
                        f'avg_batch_cost: {train_batch_cost / print_batch_step:.5f} s, '
                        f'avg_samples: {total_samples / print_batch_step}, '
                        f'ips: {total_samples / train_batch_cost:.5f} samples/s, '
                        f'eta: {eta_sec_format}')
                    self.logger.info(strs)
                    total_samples = 0
                    train_reader_cost = 0.0
                    train_batch_cost = 0.0
                reader_start = time.time()
                # eval iter step
                if self.valid_dataloader is not None and is_main_process() and (global_step > start_eval_step and
                                          (global_step - start_eval_step) %
                                          eval_batch_step == 0):
                    self.eval_step(global_step, epoch)
                # save iter step
                if is_main_process(
                ) and global_step > start_save_iter and global_step % save_iter_step == 0:
                    save_ckpt(
                        self.model,
                        self.cfg,
                        self.optimizer,
                        self.lr_scheduler,
                        epoch,
                        global_step,
                        self.best_metric,
                        is_best=False,
                        prefix=
                        f'iter_{last_whole_epoch_global_step}_{global_step}')

            # eval epoch step
            if self.valid_dataloader is not None and is_main_process() and epoch > start_eval_epoch and (
                    epoch - start_eval_epoch) % eval_epoch_step == 0:
                self.eval_step(global_step, epoch, log_samples=True)

            if is_main_process():
                save_ckpt(self.model,
                          self.cfg,
                          self.optimizer,
                          self.lr_scheduler,
                          epoch,
                          global_step,
                          self.best_metric,
                          is_best=False,
                          prefix=None)
                if epoch > start_save_epoch and (
                        epoch - start_save_epoch) % save_epoch_step == 0:
                    save_ckpt(self.model,
                              self.cfg,
                              self.optimizer,
                              self.lr_scheduler,
                              epoch,
                              global_step,
                              self.best_metric,
                              is_best=False,
                              prefix='epoch_' + str(epoch))
            last_whole_epoch_global_step = global_step
        best_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in self.best_metric.items()])}"
        self.logger.info(best_str)
        # 종료 cleanup: inner-loop의 마지막 DDP allreduce가 이미 sync point이고,
        # 그 이후 rank 0가 도는 eval_step/save_ckpt는 self.model.module로 DDP
        # 우회 → NCCL collective 없음. 따라서 추가 barrier 불필요.
        # 오히려 barrier 자체가 NCCL ALLREDUCE라, rank 0의 wandb Table 빌드 +
        # log IPC + save가 길어지는 동안 rank 1이 barrier에서 600s watchdog
        # timeout으로 죽으면서 hang을 유발함 → barrier 제거, destroy만 남김
        # (destroy_process_group은 local cleanup이라 rank별 독립 호출 안전).
        if torch.cuda.device_count() > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if self.writer is not None:
            self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
        # persistent_workers=True + pin_memory 조합에서, 인터프리터 종료 시점에
        # DataLoader 워커/프리페치 스레드가 정리되길 기다리며 hang하는 경우가
        # 잦음. _eval_train_batch가 만든 persistent iterator도 동일 이슈.
        # 명시적으로 shutdown 해서 깨끗하게 종료.
        if hasattr(self, '_train_eval_iter'):
            try:
                self._train_eval_iter._shutdown_workers()
            except Exception:
                pass
            del self._train_eval_iter
        for _attr in ('valid_dataloader', 'train_dataloader'):
            _dl = getattr(self, _attr, None)
            if _dl is None:
                continue
            _it = getattr(_dl, '_iterator', None)
            if _it is not None:
                try:
                    _it._shutdown_workers()
                except Exception:
                    pass
            setattr(self, _attr, None)
        import gc
        gc.collect()

    def _eval_train_batch(self):
        """Run one train batch in eval mode to get train acc for models like MDiff4STR."""
        if not hasattr(self, '_train_eval_iter'):
            self._train_eval_iter = iter(self.train_dataloader)
        try:
            batch = next(self._train_eval_iter)
        except StopIteration:
            self._train_eval_iter = iter(self.train_dataloader)
            batch = next(self._train_eval_iter)
        batch_tensor = [t.to(self.device) for t in batch]
        batch_numpy = [t.numpy() for t in batch]
        # DDP wrapper 우회: rank 0만 forward를 돌리므로 module로 직접 호출
        # (DDP는 모든 rank가 같은 forward 호출을 가정하므로 single-rank forward가
        #  이후 backward의 allreduce를 desync 시킴 → NCCL watchdog timeout)
        eval_model = self.model.module if hasattr(self.model, 'module') else self.model
        eval_model.eval()
        with torch.inference_mode():
            preds = eval_model(batch_tensor[0])
        preds_np = preds.detach().cpu().numpy()
        post_result = self.post_process_class(preds_np, batch_numpy, training=True)
        self.eval_class(post_result, batch_numpy, training=True)
        metric = self.eval_class.get_metric()
        eval_model.train()
        return metric

    def eval_step(self, global_step, epoch, save_best=True, log_samples=False):
        # Train acc for models that can't compute it during training (e.g. MDiff4STR)
        if self.cfg['Global'].get('cal_metric_during_train', False) and hasattr(self, 'train_dataloader'):
            train_metric = self._eval_train_batch()
            if self.wandb_run is not None:
                _payload = {f'train/{k}': v for k, v in train_metric.items()}
                _payload['global_step'] = global_step
                self.wandb_run.log(_payload)

        cur_metric = self.eval()
        wandb_samples = cur_metric.pop('_wandb_samples', [])
        cur_metric_str = f"cur metric, {', '.join(['{}: {}'.format(k, v) for k, v in cur_metric.items()])}"
        self.logger.info(cur_metric_str)

        # logger metric
        if self.writer is not None:
            for k, v in cur_metric.items():
                if isinstance(v, (float, int)):
                    self.writer.add_scalar(f'EVAL/{k}', cur_metric[k],
                                           global_step)

        # wandb logging
        if self.wandb_run is not None:
            import wandb
            wandb_log = {
                'val_acc': cur_metric.get('acc', 0),
                'val_ned': cur_metric.get('norm_edit_dis', 0),
                'epoch': epoch,
            }
            # Sample tables are heavy artifacts (each Table version uploads
            # images + metadata). Only build/log them at epoch boundaries +
            # initial eval — not every intra-epoch batch eval — so artifact
            # count stays bounded by epoch_num instead of total_eval_steps.
            if log_samples:
                # val sample table
                if wandb_samples:
                    columns = ['image', 'gt', 'pred']
                    table = wandb.Table(columns=columns)
                    for s in wandb_samples:
                        if s is None:
                            continue
                        img_np = self._denorm_image(s['image'])
                        table.add_data(
                            wandb.Image(img_np),
                            s['gt'],
                            s['pred'],
                        )
                    wandb_log['val_samples'] = table

                # train sample table (shows orig_gt vs replaced_gt vs pred)
                train_table = self._make_train_sample_table()
                if train_table is not None:
                    wandb_log['train_samples'] = train_table
            wandb_log['global_step'] = global_step
            self.wandb_run.log(wandb_log)

        if save_best and (cur_metric[self.eval_class.main_indicator] >=
                self.best_metric[self.eval_class.main_indicator]):
            self.best_metric.update(cur_metric)
            self.best_metric['best_epoch'] = epoch

            if self.writer is not None:
                self.writer.add_scalar(
                    f'EVAL/best_{self.eval_class.main_indicator}',
                    self.best_metric[self.eval_class.main_indicator],
                    global_step,
                )

            if self.wandb_run is not None:
                import wandb
                self.wandb_run.summary['best_acc'] = self.best_metric.get('acc', 0)
                self.wandb_run.summary['best_epoch'] = self.best_metric.get('best_epoch', 0)

            save_ckpt(self.model,
                      self.cfg,
                      self.optimizer,
                      self.lr_scheduler,
                      epoch,
                      global_step,
                      self.best_metric,
                      is_best=True,
                      prefix=None)
        best_str = f"best metric, {', '.join(['{}: {}'.format(k, v) for k, v in self.best_metric.items()])}"
        self.logger.info(best_str)

    @staticmethod
    def _denorm_image(img_tensor):
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        img = (img_tensor * std + mean).clamp(0, 1)
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    def _make_train_sample_table(self, n_samples=4):
        """Sample from training data: show orig_gt, replaced_gt, and model pred."""
        if self.train_dataloader is None:
            return None
        import wandb

        dataset = self.train_dataloader.dataset
        if not hasattr(dataset, 'lmdb_sets') or not hasattr(dataset, 'label_overrides'):
            return None

        # Pick random indices from training data
        n_total = len(dataset)
        indices = random.sample(range(n_total), min(n_samples, n_total))

        columns = ['image', 'gt', 'replaced_gt', 'pred']
        table = wandb.Table(columns=columns)

        # DDP wrapper 우회: rank 0만 실행하므로 forward 호출 desync 방지
        eval_model = self.model.module if hasattr(self.model, 'module') else self.model
        eval_model.eval()
        with torch.no_grad():
            for idx in indices:
                # Get a sample via the dataset
                lmdb_idx, file_idx = dataset.data_idx_order_list[idx]
                lmdb_idx, file_idx = int(lmdb_idx), int(file_idx)
                txn = dataset.lmdb_sets[lmdb_idx]['txn']

                # Read original label directly from LMDB
                label_key = f'label-{file_idx:09d}'.encode()
                raw_label = txn.get(label_key)
                orig_gt = raw_label.decode('utf-8') if raw_label else ''

                # Replaced gt (after label_overrides)
                override = dataset.label_overrides.get((lmdb_idx, file_idx))
                replaced_gt = override if override is not None else orig_gt

                # Get processed sample for model input.
                # `__getitem__` accepts [img_width, img_height, idx, ratio];
                # img_width is unused inside, but img_height/ratio drive the
                # actual resize (base_shape[ratio-1]). Use ratio (1..max_ratio)
                # rather than the argsort index for any width-derived value.
                try:
                    ratio = int(dataset.wh_ratio[idx])
                    w = int(ratio * 32)
                    sample = dataset.__getitem__([w, 32, idx, ratio])
                    if sample is None:
                        continue
                    img_tensor = sample[0].to(self.device).unsqueeze(0)
                    preds = eval_model(img_tensor)
                    if isinstance(preds, (list, tuple)):
                        preds = preds[-1]
                    preds_np = preds.detach().cpu().float().numpy()
                    decoded = self.post_process_class(preds_np)
                    # ARLabelDecode without batch returns [(text, conf), ...];
                    # GTC-style returns a list — take the first head.
                    if isinstance(decoded, list) and decoded and isinstance(decoded[0], list):
                        decoded = decoded[0]
                    pred_text = decoded[0][0] if decoded else ''
                    if self.pl_charset_adapter:
                        pred_text = self.pl_charset_adapter(pred_text)

                    img_np = self._denorm_image(sample[0])
                    table.add_data(
                        wandb.Image(img_np),
                        orig_gt,
                        replaced_gt,
                        pred_text,
                    )
                except Exception as e:
                    if not getattr(self, '_train_table_warned', False):
                        self.logger.warning(
                            f'_make_train_sample_table sample failed '
                            f'(lmdb_idx={lmdb_idx}, file_idx={file_idx}): {e!r}')
                        self._train_table_warned = True
                    continue
        eval_model.train()
        return table if len(table.data) > 0 else None

    def eval(self, error_save_dir=None, dataset_name=None,
             predictions_save_path=None, no_gt=False):
        # DDP wrapper 우회: rank 0만 eval을 돌리기 때문에 wrapped forward를 쓰면
        # rank 0/1의 forward 호출 횟수가 어긋나 이후 training step의 allreduce가
        # hang → NCCL watchdog timeout.
        eval_model = self.model.module if hasattr(self.model, 'module') else self.model
        eval_model.eval()
        # Reservoir sampling: collect 4 random samples for wandb table
        wandb_samples = []
        wandb_sample_k = 4
        sample_counter = 0
        error_img_idx = 0
        prediction_rows = [] if predictions_save_path is not None else None
        # Detect file_idx position in keep_keys so we can split it off before
        # passing batch to post_process (some PostProcess classes slice by
        # position, e.g. GTCLabelDecode, and would otherwise break).
        keep_keys = (self.cfg.get('Eval', {}).get('dataset', {})
                     .get('transforms', [{}])[-1]
                     .get('KeepKeys', {}).get('keep_keys', []))
        file_idx_pos = (len(keep_keys) - 1 - keep_keys[::-1].index('file_idx')
                        if 'file_idx' in keep_keys else None)

        with torch.no_grad():
            total_frame = 0.0
            total_time = 0.0
            pbar = tqdm(
                total=len(self.valid_dataloader),
                desc='eval model:',
                position=0,
                leave=True,
            )
            sum_images = 0
            for idx, batch in enumerate(self.valid_dataloader):
                batch_tensor = [t.to(self.device) for t in batch]
                batch_numpy = [t.numpy() for t in batch]
                # Split file_idx off from batch before feeding downstream
                file_idxs = None
                if file_idx_pos is not None and file_idx_pos < len(batch_numpy):
                    file_idxs = batch_numpy[file_idx_pos]
                    batch_numpy = (batch_numpy[:file_idx_pos]
                                   + batch_numpy[file_idx_pos + 1:])
                    batch_tensor = (batch_tensor[:file_idx_pos]
                                    + batch_tensor[file_idx_pos + 1:])
                start = time.time()
                if self.scaler:
                    with torch.cuda.amp.autocast(
                            enabled=self.device.type == 'cuda'):
                        preds = eval_model(batch_tensor[0],
                                           data=batch_tensor[1:])
                else:
                    preds = eval_model(batch_tensor[0], data=batch_tensor[1:])

                total_time += time.time() - start
                # Obtain usable results from post-processing methods
                # Evaluate the results of the current batch
                post_result = self.post_process_class(preds, batch_numpy)
                # CCD mode: map extended Unicode chars back to base chars
                raw_preds_list = None
                if self.pl_charset_adapter is not None and isinstance(post_result, tuple):
                    preds_list, labels_list = post_result
                    raw_preds_list = preds_list  # before adapter
                    preds_list = [(self.pl_charset_adapter(text), conf) for text, conf in preds_list]
                    post_result = (preds_list, labels_list)
                if not no_gt:
                    self.eval_class(post_result, batch_numpy)

                # Save error images / all predictions (same comparison as metric)
                # GTC decoder returns [gtc_tuple, ctc_tuple]; RecGTCMetric reports
                # CTC as the primary 'acc', so save from the CTC branch.
                per_sample_source = None
                if isinstance(post_result, tuple):
                    per_sample_source = post_result
                elif (isinstance(post_result, list) and len(post_result) == 2
                      and isinstance(post_result[1], tuple)):
                    per_sample_source = post_result[1]
                need_per_sample = (
                    (error_save_dir is not None or prediction_rows is not None)
                    and per_sample_source is not None)
                if need_per_sample:
                    cur_preds, cur_labels = per_sample_source
                    images = batch_tensor[0].cpu() if error_save_dir is not None else None
                    # file_idxs was split off from batch before post_process
                    # Get metric comparison settings
                    _eval = self.eval_class
                    if hasattr(_eval, 'ctc_metric'):
                        _eval = _eval.ctc_metric  # RecGTCMetric wrapper
                    _norm = _eval._normalize_text if hasattr(_eval, '_normalize_text') else (lambda x: x)
                    _lower = getattr(_eval, 'is_lower', True)
                    _ignore_space = getattr(_eval, 'ignore_space', True)
                    def _cmp(t):
                        if _ignore_space:
                            t = t.replace(' ', '')
                        t = _norm(t)
                        if _lower:
                            t = t.lower()
                        return t
                    for i in range(len(cur_preds)):
                        pred_text = cur_preds[i][0]
                        gt_text = cur_labels[i]
                        if isinstance(gt_text, (tuple, list)):
                            gt_text = gt_text[0]
                        img_idx = int(file_idxs[i]) if file_idxs is not None else -1
                        # Match metric comparison: normalize + lower + ignore_space
                        is_correct = (False if no_gt
                                      else _cmp(pred_text) == _cmp(gt_text))
                        if prediction_rows is not None:
                            gt_out = '' if no_gt else gt_text
                            correct_out = '' if no_gt else int(is_correct)
                            prediction_rows.append(
                                (img_idx, gt_out, pred_text, correct_out))
                        if error_save_dir is not None and not no_gt and not is_correct:
                            save_idx = img_idx if img_idx >= 0 else error_img_idx
                            self._save_error_image(
                                images[i], error_save_dir, dataset_name,
                                save_idx, pred_text, gt_text)
                            error_img_idx += 1

                # Reservoir sampling for wandb table
                if self.wandb_run is not None and isinstance(post_result, tuple):
                    cur_preds, cur_labels = post_result
                    images = batch_tensor[0].cpu()
                    for i in range(len(cur_preds)):
                        sample_counter += 1
                        # Decide whether to include this sample
                        if len(wandb_samples) < wandb_sample_k:
                            slot = len(wandb_samples)
                            wandb_samples.append(None)
                        else:
                            slot = random.randint(0, sample_counter - 1)
                            if slot >= wandb_sample_k:
                                continue
                        gt = cur_labels[i]
                        if isinstance(gt, (tuple, list)):
                            gt = gt[0]
                        wandb_samples[slot] = {
                            'image': images[i],
                            'gt': gt,
                            'pred': cur_preds[i][0],
                        }

                pbar.update(1)
                total_frame += len(batch[0])
                sum_images += 1
            # Get final metric，eg. acc or hmean
            if no_gt:
                metric = {}
            else:
                metric = self.eval_class.get_metric()

        if error_save_dir is not None:
            self.logger.info(
                f'Saved {error_img_idx} error images to {error_save_dir}')
        if prediction_rows is not None:
            import csv
            os.makedirs(os.path.dirname(predictions_save_path) or '.',
                        exist_ok=True)
            with open(predictions_save_path, 'w', newline='',
                      encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['file_idx', 'gt', 'pred', 'correct'])
                w.writerows(prediction_rows)
            self.logger.info(
                f'Saved {len(prediction_rows)} predictions to {predictions_save_path}')
        pbar.close()
        eval_model.train()
        metric['fps'] = total_frame / total_time if total_time > 0 else 0.0
        metric['_wandb_samples'] = wandb_samples
        return metric

    @staticmethod
    def _save_error_image(img_tensor, save_dir, dataset_name, img_idx,
                          pred_text, gt_text):
        import re
        from PIL import Image

        def _sanitize(s):
            if isinstance(s, (tuple, list)):
                s = s[0]
            return re.sub(r'[\\/:*?"<>|]', '_', str(s))

        fname = f'{dataset_name}_{img_idx:04d}_pred_{_sanitize(pred_text)}_gt_{_sanitize(gt_text)}.png'
        # tensor: (C, H, W), normalized to [0, 1] or [-1, 1]
        img = img_tensor.clone()
        if img.min() < 0:
            img = (img + 1) / 2
        img = img.clamp(0, 1)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
        Image.fromarray(img_np).save(os.path.join(save_dir, fname))

    def test_dataloader(self):
        starttime = time.time()
        count = 0
        try:
            for data in self.train_dataloader:
                count += 1
                if count % 1 == 0:
                    batch_time = time.time() - starttime
                    starttime = time.time()
                    self.logger.info(
                        f'reader: {count}, {data[0].shape}, {batch_time}')
        except:
            import traceback

            self.logger.info(traceback.format_exc())
        self.logger.info(f'finish reader: {count}, Success!')
