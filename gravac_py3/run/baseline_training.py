import time
import logging
import numpy as np
import os
import argparse

import torch
import torch.distributed as dist
from torch.autograd import Variable

import gravac_py3.helper.misc as misc
from gravac_py3.helper import dataloader
from gravac_py3.models import lstm_model
import gravac_py3.helper.communication as comm


class DistributedTraining(object):
    def __init__(self, args):
        self.args = args
        self.world_size = args.world_size
        self.trainer_rank = args.rank
        self.train_bsz = args.train_bsz
        self.val_bsz = args.val_bsz
        self.global_bsz = self.train_bsz * self.world_size
        self.dist_backend = args.dist_backend
        logging.basicConfig(filename=args.log_dir + 'trainer-' + str(self.trainer_rank) + '.log',
                            level=logging.INFO)
        dist.init_process_group(self.dist_backend, rank=self.trainer_rank, world_size=self.world_size)

        self.model_name = args.model_name
        self.dataset = args.dataset
        self.compress_ratio = float(args.compression_ratio)

        self.modelobj = misc.get_model(self.model_name, self.args, self.world_size)
        self.loss_fn = self.modelobj.get_loss()
        self.optimizer = self.modelobj.get_optimizer()
        self.lr_schedule = self.modelobj.get_lrschedule()
        self.global_step = 0
        self.losses, self.top1, self.topx = None, None, None
        # async mode set to False for CUDA streams
        self.async_op = False
        self.do_compression = False if args.do_compression == 0 else self.do_compression = True

        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(self.trainer_rank))
        else:
            self.device = torch.device("cpu")

        self.model = self.modelobj.get_model().to(self.device)
        self.trainloader, self.vocab_size, self.dataset_size = dataloader.TrainingData(self.args, self.train_bsz,
                                                                                       self.trainer_rank).train_data()
        self.test_loader, self.vocab_size = dataloader.TestData(self.args).test_data()

        self.compressor = misc.CompressionType(compression=args.compression, device=self.device,
                                               compress_ratio=self.compress_ratio).get_compressor()
        self.windowsize = args.windowsize
        self.alpha = args.alpha
        self.model_helper = misc.ModelHelper(model=self.model, compressor=self.compressor)
        self.commops = comm.CollectiveCommOps(self.world_size, self.async_op, self.model_helper, self.device)
        self.num_steps = args.num_steps
        self.eval_steps = args.eval_steps

    def test_model(self):
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
                    logging.info(f"VALIDATION METRICS logged on step %d lossval %f testloss %f top1val %f top1exp %f",
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

    def launch_training(self):
        self.model = self.commops.broadcast(self.model, rank=0)
        track_grads = misc.MovingWindowAverage(window_size=self.windowsize, alpha=self.alpha)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

        self.model_helper.count_model_parameters()
        if self.model_name == 'lstm':
            hidden = self.model.init_hidden()

        for epoch in range(self.args.epochs):
            self.losses, self.top1, self.topx = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
            for _, record in enumerate(self.trainloader, 0):
                inputs, labels = record
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.model_name == 'lstm':
                    inputs = Variable(inputs.transpose(0, 1).contiguous()).to(self.device)
                    labels = Variable(labels.transpose(0, 1).contiguous()).to(self.device)
                    hidden = lstm_model.repackage_hidden(hidden)

                compute_strt = time.time()
                if self.model_name == 'lstm':
                    outputs, hidden = self.model(inputs, hidden)
                    tt = torch.squeeze(labels.view(-1, self.train_bsz * self.num_steps))
                    loss = self.loss_fn(outputs.view(-1, self.vocab_size), tt)
                else:
                    output = self.model(inputs)
                    loss = self.loss_fn(output, labels)

                loss.backward()
                compute_time = time.time() - compute_strt
                self.global_step += 1

                if self.global_step > 0 and self.global_step % self.args.trainacc_step == 0:
                    if self.model_name != 'lstm':
                        self.train_accuracy(inputs=inputs, labels=labels, output=output, loss=loss,
                                            step=self.global_step, epoch=epoch)

                gradients = [p.grad for p in self.model.parameters()]

                # perform compression when do_compression is True else call allreduce on uncompressed tensors
                if self.do_compression:
                    misc.sync_device(self.device)
                    compress_strt = time.time()
                    layer_values, layer_indices = self.model_helper.gradient_compression(gradients, self.compress_ratio)
                    misc.sync_device(self.device)
                    compress_time = time.time() - compress_strt

                    effective_ratio = self.model_helper.effective_ratio(tensors_compressed=layer_values,
                                                                        tensors_uncompressed=gradients, device=self.device)

                    misc.sync_device(self.device)
                    comm_start = time.time()
                    reduced_grads = self.commops.compression_allgather(layer_values, layer_indices)
                    for p, g in zip(self.model.parameters(), reduced_grads):
                        p.grad = g

                    misc.sync_device(self.device)
                    comm_time = time.time() - comm_start

                    logging.info("DECENTRALIZED STATIC COMPRESSION epoch %d itr %d computation_time %f compress_time %f "
                                 "communication_time %f effective_ratio_achieved %f", epoch, self.global_step,
                                 compute_time, compress_time, comm_time, effective_ratio)

                else:
                    # baseline allreduce without performing any compression
                    misc.sync_device(self.device)
                    comm_strt = time.time()
                    self.model = self.commops.allreduce(self.model)
                    misc.sync_device(self.device)
                    comm_time = time.time() - comm_strt

                    logging.info('BASELINE ALLREDUCE epoch %d itr %d computation_time %f communication_time %f', epoch,
                                 self.global_step, compute_time, comm_time)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.test_model()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='29501')
    parser.add_argument('--seed', type=int, default=1234, help='seed value for result replication')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='# training steps after which test model performance')
    parser.add_argument('--log-dir', type=str, default='/', help='dir location when data is saved')
    parser.add_argument('--train-dir', type=str, default='/')
    parser.add_argument('--test-dir', type=str, default='/')
    parser.add_argument('--world-size', type=int, default=1, help='# overall procs to spawn across cluster')
    parser.add_argument('--rank', type=int, help='process rank', default=0)
    parser.add_argument('--epochs', type=int, help='# epochs to run', default=300)
    parser.add_argument('--compression', type=str, default='topK')
    parser.add_argument('--compression-ratio', type=float, default=0.1)
    parser.add_argument('--compress-stages', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--test-bsz', type=int, default=128)
    parser.add_argument('--train-bsz', type=int, default=32)
    parser.add_argument('--trainacc-step', type=int, default=50)
    parser.add_argument('--do-compression', type=int, default=0, help='train without or with compression')
    parser.add_argument('--num-steps', type=int, default=35)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model-name', type=str, default='resnet101')
    parser.add_argument('--windowsize', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.02)
    parser.add_argument('--dist-backend', type=str, default='gloo')

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

