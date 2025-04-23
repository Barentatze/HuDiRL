import copy
import csv
import functools
import os

import random
import time

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from mpi4py import MPI

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import ImageReward as RewardModel
import torchvision.utils as vutils
import torchvision as tv
from guided_diffusion.critic import RewardPredictor

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            using_rl=True,
            alpha=0.1,
            rl_Pool2dSize=16,
            # rl_H=128,
            # rl_W=128,
    ):
        self.critic_losses_file = "critic_losses.csv"
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.RL = using_rl
        self.alpha = alpha
        self.timestep = 50
        # self.rl_H, self.rl_W = rl_H, rl_W
        self.critic = RewardPredictor(Pool2dSize=rl_Pool2dSize)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.reward_model = RewardModel.load("ImageReward-v1.0")
        self.prompt = "a building on stairs"

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        # self._anneal_lr()
        self.log_step()

    def is_rank0(self):
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        self.critic.to(dist_util.dev())
        self.critic.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # random resize
            if MPI.COMM_WORLD.Get_rank() == 0:
                curr_h = round(micro.shape[2] * random.uniform(0.75, 1.25))
                curr_w = round(micro.shape[3] * random.uniform(0.75, 1.25))
                curr_h, curr_w = 4 * (curr_h // 4), 4 * (curr_w // 4)
                MPI.COMM_WORLD.bcast((curr_h, curr_w))
            else:
                curr_h, curr_w = MPI.COMM_WORLD.bcast(None)
            micro = F.interpolate(micro, (curr_h, curr_w), mode="bicubic")

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            if self.RL:
                self.critic.eval()
                start_critic_time = time.time()
                t = th.tensor([self.timestep], device=dist_util.dev())
                # x_t = th.randn((1, 3, self.rl_H, self.rl_W), device=dist_util.dev())
                x_t = th.randn((1, 3, curr_h, curr_w), device=dist_util.dev())
                model_kwargs = {}

                with th.no_grad():
                    model_output = self.ddp_model(x_t, t, **model_kwargs)  # Generate a diffusion step

                # Using Critic to evaluate
                with th.no_grad():
                    reward = self.critic(model_output)
                end_critic_time = time.time()
                # reward = th.tensor(reward).to(dist_util.dev())
                if self.is_rank0:
                    print(f"Step {self.step} - Critic Reward: {reward.item():.4f}, Time for critic: {end_critic_time - start_critic_time:.4f} seconds")

                loss = loss - self.alpha * reward
                # loss = (1 - self.alpha) * loss + self.alpha * reward

                if self.step % 100 == 0: # train the critic every 100 steps
                    self.critic.train()
                    
                    critic_losses = []
                    for i in range(10): # train step
                        start_training_time = time.time()
                        # x_t = th.randn((1, 3, self.rl_H, self.rl_W), device=dist_util.dev())
                        x_t = th.randn((1, 3, curr_h, curr_w), device=dist_util.dev())
                        with th.no_grad():
                            model_output = self.ddp_model(x_t, t, **model_kwargs)

                        predicted_reward = self.critic(model_output)

                        # Generate and save a complete image
                        with th.no_grad():
                            sample = self.diffusion.p_sample_loop(
                                self.ddp_model,
                                # (1, 3, self.rl_H, self.rl_W),
                                (1, 3, curr_h, curr_w),
                                model_kwargs=model_kwargs,
                                device=dist_util.dev(),
                                progress=False,
                                noise=x_t
                            )
                            x_gen = sample[0]
                        os.makedirs("generated_train_imgs", exist_ok=True)
                        # save_path = f"generated_train_imgs/step_{self.step}_{i}.png"
                        rank = dist.get_rank() if dist.is_initialized() else 0
                        save_path = f"generated_train_imgs/step_{self.step}_{i}_rank{rank}.png"
                        vutils.save_image(x_gen * 0.5 + 0.5, save_path)

                        # Get the actual reward
                        actual_reward = self.reward_model.score(self.prompt, save_path)
                        actual_reward = th.tensor(actual_reward, dtype=th.float32, device=dist_util.dev())

                        end_training_time = time.time()

                        critic_loss = F.mse_loss(predicted_reward, actual_reward)
                        critic_losses.append(critic_loss.item())

                        if self.is_rank0:
                            print(f"Train Step {self.step}_{i} - Critic Loss: mse({actual_reward.item():.4f} - {predicted_reward.item():.4f}) = {critic_loss.item():.4f}, Time for training: {end_training_time - start_training_time:.4f} seconds")

                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        self.critic_optimizer.step()
                    
                    # Record the critic loss
                    avg_critic_loss = sum(critic_losses) / len(critic_losses)
                    with open(self.critic_losses_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.step, avg_critic_loss])

                    self.critic.eval()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

            if self.step % 1000 == 0:  # show the evaluation results every 1000 steps
                for i in range(5):  # train step
                    model_kwargs={}
                    # Generate and save a complete image
                    with th.no_grad():
                        sample = self.diffusion.p_sample_loop(
                            self.ddp_model,
                            (1, 3, curr_h, curr_w),
                            model_kwargs=model_kwargs,
                            device=dist_util.dev(),
                            progress=False,
                        )
                        x_gen = sample[0]
                    os.makedirs(f"evaluation_imgs_RL_{self.RL}", exist_ok=True)
                    save_path = f"evaluation_imgs_RL_{self.RL}/step_{self.step}_{i}.png"
                    vutils.save_image(x_gen * 0.5 + 0.5, save_path)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
                    "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
