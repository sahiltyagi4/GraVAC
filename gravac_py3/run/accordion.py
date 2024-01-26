import logging
import time
import numpy as np
import argparse
import os

import torch
import torch.distributed as dist
from torch.autograd import Variable

import gravac_py3.helper.misc as misc
from gravac_py3.models import lstm_model
from gravac_py3.helper import dataloader
import gravac_py3.helper.communication as comm


class AccordianTraining(object):
    def __init__(self, args):
        self.args = args
        self.world_size = args.world_size
        self.trainer_rank = args.rank
        self.tid = args.tid
        self.train_bsz = args.train_bsz
        self.val_bsz = args.val_bsz
        self.global_bsz = self.train_bsz * self.world_size
        self.dist_backend = args.dist_backend
        logging.basicConfig(filename=self.args.log_dir + 'trainer-' + str(self.trainer_rank) + '.log', level=logging.INFO)
        dist.init_process_group(self.dist_backend, rank=self.trainer_rank, world_size=self.world_size)

        self.model_name = args.model_name
        self.dataset = args.dataset

        self.modelobj = misc.get_model(self.model_name, self.args, self.world_size)
        self.loss_fn = self.modelobj.get_loss()
        self.optimizer = self.modelobj.get_optimizer()
        self.lr_schedule = self.modelobj.get_lrschedule()
        self.global_step = 0
        self.losses, self.top1, self.topx = None, None, None
        # async mode set to False for CUDA streams
        self.async_op = False

        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(self.trainer_rank))
        else:
            self.device = torch.device("cpu")

        print(f'device name is {self.device}')

        self.model = self.modelobj.get_model().to(self.device)
        self.trainloader, self.vocab_size, self.dataset_size = dataloader.TrainingData(self.args, self.train_bsz,
                                                                                       self.trainer_rank).train_data()
        self.test_loader, self.vocab_size = dataloader.TestData(self.args).test_data()
        self.compressor = misc.CompressionType(compression=args.compression, device=self.device,
                                               compress_ratio=self.low_cf).get_compressor()

        self.model_helper = misc.ModelHelper(model=self.model, compressor=self.compressor)
        self.commops = comm.CollectiveCommOps(self.world_size, self.async_op, self.model_helper, self.device)
        self.num_steps = self.args.num_steps
        self.eval_steps = self.args.eval_steps

        self.windowsize = self.args.window_size
        self.alpha = self.args.alpha
        self.low_cf = self.args.low_cf
        self.high_cf = self.args.high_cf
        self.norm_threshold  = self.args.norm_threshold

    def test_model(self):
        with torch.no_grad():
            if self.model_name == 'lstm':
                if self.global_step > self.eval_steps and self.global_step % self.eval_steps == 0:
                    test_loss, costs, total_steps, total_iters, total = 0.0, 0.0, 0, 0, 0.0
                    costs = 0.0
                    with torch.no_grad():
                        for _, record in enumerate(self.test_loader, 0):
                            inputs, labels = record
                            inputs = Variable(inputs.transpose(0, 1).contiguous()).to(self.device)
                            labels = Variable(labels.transpose(0, 1).contiguous()).to(self.device)
                            hidden = self.model.init_hidden()
                            hidden = lstm_model.repackage_hidden(hidden)
                            outputs, hidden = self.model(inputs, hidden)
                            tt = torch.squeeze(labels.view(-1, self.val_bsz * self.num_steps))
                            loss = self.loss_fn(outputs.view(-1, self.vocab_size), tt)
                            test_loss += loss.item()
                            costs += loss.item() * self.num_steps
                            total_steps += self.num_steps
                            total += labels.size(0)
                            total_iters += 1

                        test_loss /= total_iters
                        acc_val = costs / total_steps
                        acc = np.exp(acc_val)
                        loss = float(test_loss) / total
                        logging.info(
                            f"VALIDATION METRICS logged on step %d lossval %f testloss %f top1val %f top1exp %f",
                            self.global_step, loss, test_loss, acc_val, acc)
            else:
                if self.global_step > self.eval_steps and self.global_step % self.eval_steps == 0:
                    top1, top10, losses = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
                    with torch.no_grad():
                        for _, record in enumerate(self.test_loader, 0):
                            inputs, labels = record
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            output = self.model(inputs)
                            loss = self.loss_fn(output, labels)
                            if isinstance(output, tuple):
                                print(f"output is tuple type")
                                loss = sum(self.loss_fn(out, labels) for out in output)
                                for o in output:
                                    prec1 = misc.test_accuracy(o.data, labels, topk=(1,))
                                    top1.update(prec1[0], inputs.size(0))
                                    prec10 = misc.test_accuracy(o.data, labels, topk=(10,))
                                    top10.update(prec10[0], inputs.size(0))

                                losses.update(loss.data[0], inputs.size(0) * len(output))
                            else:
                                prec1 = misc.test_accuracy(output.data, labels, topk=(1,))
                                top1.update(prec1[0], inputs.size(0))
                                prec10 = misc.test_accuracy(output.data, labels, topk=(10,))
                                top10.update(prec10[0], inputs.size(0))
                                losses.update(loss.item(), inputs.size(0))

                        logging.info(f"VALIDATION METRICS logged on step %d lossval %f lossavg %f top1val %f top1avg %f "
                                     f"top10val %f top10avg %f", self.global_step, losses.val, losses.avg,
                                     top1.val.cpu().numpy().item(), top1.avg.cpu().numpy().item(),
                                     top10.val.cpu().numpy().item(), top10.avg.cpu().numpy().item())

    def train_accuracy(self, inputs, labels, output, loss, step, epoch):
        if self.model_name != 'lstm':
            with torch.no_grad():
                prec1 = misc.test_accuracy(output.data, labels, topk=(1,))

                if self.dataset == 'cifar10':
                    precx = misc.test_accuracy(output.data, labels, topk=(5,))
                elif self.dataset == 'cifar100':
                    precx = misc.test_accuracy(output.data, labels, topk=(10,))
                elif self.dataset == 'imagenet':
                    precx = misc.test_accuracy(output.data, labels, topk=(10,))

                self.top1.update(prec1[0], inputs.size(0))
                self.topx.update(precx[0], inputs.size(0))
                self.losses.update(loss.item(), inputs.size(0))
                logging.info('TRAINING_METRICS logged at step %d epoch %d lossval %f lossavg %f top1val %f top1avg %f '
                             'top10val %f top10avg %f', step, epoch, self.losses.val, self.losses.avg,
                             self.top1.val.cpu().numpy().item(), self.top1.avg.cpu().numpy().item(),
                             self.topx.val.cpu().numpy().item(), self.topx.avg.cpu().numpy().item())

    def change(self, curr, prev):
        if prev > 0.:
            return abs(curr - prev) / prev
        else:
            return 0.

    def launch_training(self):
        self.model = self.commops.broadcast(self.model, rank=0)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

        self.model_helper.count_model_parameters()
        if self.model_name == 'lstm':
            hidden = self.model.init_hidden()

        track_grads = misc.MovingWindowAverage(window_size=self.windowsize, alpha=self.alpha)
        cf = self.low_cf
        prev_norm = 0.

        for epoch in range(self.args.epochs):
            self.losses, self.top1, self.topx = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
            for _, record in enumerate(self.trainloader, 0):
                inputs, labels = record
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.global_step += 1

                begin = time.time()
                if self.model_name == 'lstm':
                    inputs = Variable(inputs.transpose(0, 1).contiguous()).to(self.device)
                    labels = Variable(labels.transpose(0, 1).contiguous()).to(self.device)
                    hidden = lstm_model.repackage_hidden(hidden)
                    outputs, hidden = self.model(inputs, hidden)
                    tt = torch.squeeze(labels.view(-1, self.train_bsz * self.num_steps))
                    loss = self.loss_fn(outputs.view(-1, self.vocab_size), tt)
                else:
                    output = self.model(inputs)
                    loss = self.loss_fn(output, labels)

                loss.backward()
                compute_time = time.time() - begin

                if self.global_step > 0 and self.global_step % self.args.trainacc_step == 0:
                    if self.model_name != 'lstm':
                        self.train_accuracy(inputs=inputs, labels=labels, output=output, loss=loss,
                                            step=self.global_step, epoch=epoch)

                grads = [p.grad for p in self.model.parameters()]
                curr_norm = track_grads.compute_moving_avg(self.model_helper.grad_norm(grads))
                delta_change = self.change(curr_norm, prev_norm)
                cf = self.low_cf if delta_change >= self.norm_threshold else self.high_cf

                misc.sync_device(self.device)
                begin = time.time()
                layer_values, layer_indices = self.model_helper.gradient_compression(grads, cf)
                misc.sync_device(self.device)
                compress_time = time.time() - begin

                misc.sync_device(self.device)
                begin = time.time()
                compressed_grads = self.commops.compression_allgather(layer_values, layer_indices)
                for p, g in zip(self.model.parameters(), compressed_grads):
                    p.grad = g

                misc.sync_device(self.device)
                comm_time = time.time() - begin

                prev_norm = curr_norm
                self.optimizer.step()
                self.optimizer.zero_grad()
                logging.info(f'ACCORDION COMPRESSION epoch {epoch} itr {self.global_step} compute_time {compute_time} '
                             f'compress_time {compress_time} communication_time {comm_time} with cf_used {cf}')

                self.test_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='29501')
    parser.add_argument('--seed', type=int, default=1234, help='seed value for result replication')
    parser.add_argument('--eval-steps', type=int, default=100, help='# training steps after which test model performance')
    parser.add_argument('--log-dir', type=str, default='/', help='dir location when data is saved')
    parser.add_argument('--train-dir', type=str, default='/')
    parser.add_argument('--test-dir', type=str, default='/')
    parser.add_argument('--world-size', type=int, default=1, help='# overall procs to spawn across cluster')
    parser.add_argument('--rank', type=int, help='process rank', default=0)
    parser.add_argument('--epochs', type=int, help='# epochs to run', default=300)
    parser.add_argument('--compression', type=str, default='dgc')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--test-bsz', type=int, default=128)
    parser.add_argument('--train-bsz', type=int, default=32)
    parser.add_argument('--trainacc-step', type=int, default=50)
    parser.add_argument('--num-steps', type=int, default=35)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model-name', type=str, default='resnet101')
    parser.add_argument('--windowsize', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.02)
    parser.add_argument('--dist-backend', type=str, default='gloo')
    parser.add_argument('--low-cf', type=float, default=0.99)
    parser.add_argument('--high-cf', type=float, default=0.25)
    parser.add_argument('--norm-threshold', type=float, default=0.2)
    parser.add_argument('--tid', type=int, default=1)
    parser.add_argument('--window-size', type=int, default=100)

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    AccordianTraining(args).launch_training()