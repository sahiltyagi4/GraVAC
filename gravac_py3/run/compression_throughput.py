import logging
import numpy as np
import time
import os
import argparse

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.backends.cudnn import deterministic

import gravac_py3.helper.misc as misc
from gravac_py3.helper import dataloader
from gravac_py3.models import lstm_model
import gravac_py3.helper.communication as comm


class CompressionThroughputTraining(object):
    def __init__(self, args):
        self.args = args
        self.world_size = self.args.cluster_world_size
        self.trainer_rank = self.args.rank
        self.tid = self.args.tid
        self.train_bsz = self.args.train_bsz
        self.val_bsz = self.args.val_bsz
        self.global_bsz = self.train_bsz * self.world_size
        self.dist_backend = self.args.dist_backend
        logging.basicConfig(filename=self.args.log_dir + 'trainer-' + str(self.trainer_rank) + str(self.tid) + '.log',
                            level=logging.INFO)
        dist.init_process_group(self.dist_backend, rank=self.args.rank, world_size=self.world_size)

        self.model_name = self.args.model_name
        self.dataset = self.args.dataset
        self.compression = self.args.compression

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

        self.model = self.modelobj.get_model().to(self.device)
        self.trainloader, self.vocab_size, self.dataset_size = dataloader.TrainingData(self.args, self.train_bsz,
                                                                                       self.trainer_rank).train_data()
        self.test_loader, self.vocab_size = dataloader.TestData(self.args).test_data()

        # self.compressor = misc.CompressionType(compression=self.compression, device=self.device).get_compressor()
        # self.model_helper = misc.ModelHelper(model=self.model, compressor=self.compressor)

        # need to keep different compressors for minCF and currCF to keep track of residuals for each
        self.minCF_compressor = misc.CompressionType(compression=self.compression, device=self.device).get_compressor()
        self.currCF_compressor = misc.CompressionType(compression=self.compression, device=self.device).get_compressor()

        self.minCF_helper = misc.ModelHelper(model=self.model, compressor=self.minCF_compressor)
        self.currCF_helper = misc.ModelHelper(model=self.model, compressor=self.currCF_compressor)

        self.comm_ops = comm.CollectiveCommOps(self.world_size, self.async_op, self.currCF_helper, self.device)
        self.num_steps = self.args.num_steps
        self.eval_steps = self.args.eval_steps

        # min. CF to use
        self.min_cf = float(self.args.min_cf)
        # max. CF to use
        self.max_cf = float(self.args.max_cf)
        # for multi-level compression... this is 1/m
        # CF scaling policy to double CF on each subsequent adjustment, or exponentially scaling it
        self.cfstepsize = float(self.args.cfstepsize)
        # do compression eval for next possible cf after these many steps
        self.compsteps = int(self.args.compsteps)
        # min compression efficiency epsilon
        self.epsilon = float(self.args.epsilon)
        # COMPRESSION THROUGHPUT threshold
        self.omega = float(self.args.omega)
        # for expoential weighted smoothing
        self.windowsize, self.alpha = int(self.args.windowsize), float(self.args.alpha)

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

                        logging.info(
                            f"VALIDATION METRICS logged on step %d lossval %f lossavg %f top1val %f top1avg %f "
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
        self.model = self.comm_ops.broadcast(self.model, rank=0)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

        self.minCF_helper.count_model_parameters()
        if self.model_name == 'lstm':
            hidden = self.model.init_hidden()

        curr_cf = self.min_cf
        searchCF = True
        cfs_evaluated = set()
        # keep track of minimum and current CF's compression gain
        minCFgrad_tracker = misc.MovingWindowAverage(window_size=self.windowsize, alpha=self.alpha)
        currCFgrad_tracker = misc.MovingWindowAverage(window_size=self.windowsize, alpha=self.alpha)
        bpgrad_tracker = misc.MovingWindowAverage(window_size=self.windowsize, alpha=self.alpha)

        compression_throughput = {}
        logging.info(f'training with params windowsize {self.windowsize} '
                     f'alpha {self.alpha} epsilon {self.epsilon} omega {self.omega} cfstepsize {self.cfstepsize} '
                     f'theta_min {self.min_cf} max_cf {self.max_cf}')

        logged_tput, logged_cf, logged_gain, logged_commtime, logged_endtime = 0., 0., 0., 0., 0.
        logged_str = ''

        for epoch in range(self.args.epochs):
            self.losses, self.top1, self.topx = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
            for _, record in enumerate(self.trainloader, 0):
                self.test_model()
                inputs, labels = record
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.global_step += 1

                misc.sync_device(self.device)
                begin_time = time.time()
                compute_strt = time.time()
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
                misc.sync_device(self.device)
                compute_time = time.time() - compute_strt
                if self.global_step > 0 and self.global_step % self.args.trainacc_step == 0:
                    if self.model_name != 'lstm':
                        self.train_accuracy(inputs=inputs, labels=labels, output=output, loss=loss,
                                            step=self.global_step, epoch=epoch)

                # backpropagated gradients
                bp_grads = [p.grad for p in self.model.parameters()]
                bpgrad_norm = bpgrad_tracker.compute_moving_avg(self.minCF_helper.grad_norm(bp_grads))
                # gradient values and indices corresponding to minCF
                minCF_gvals, minCF_gixs = self.minCF_helper.gradient_compression(bp_grads, self.min_cf)
                minCF_gradnorm = minCFgrad_tracker.compute_moving_avg(self.minCF_helper.grad_norm(self.minCF_helper.layerwise_compressed_grads()))
                minCF_compgain = self.minCF_helper.compression_gain(uncompressed_norm=bpgrad_norm, compressed_norm=minCF_gradnorm)
                if self.global_step >= self.windowsize:
                    if curr_cf != self.min_cf:
                        currCF_gvals, currCF_gixs = self.currCF_helper.gradient_compression(bp_grads, curr_cf)
                        currCF_gradnorm = currCFgrad_tracker.compute_moving_avg(
                            self.minCF_helper.grad_norm(self.minCF_helper.layerwise_compressed_grads()))
                        currCF_compgain = self.currCF_helper.compression_gain(uncompressed_norm=bpgrad_norm,
                                                                              compressed_norm=currCF_gradnorm)

                        if currCF_compgain >= self.epsilon:
                            # communicating gradients compressed to CF currCF
                            misc.sync_device(self.device)
                            comm_strt = time.time()
                            currCF_reducedgrads = self.comm_ops.compression_allgather(layer_values=currCF_gvals,
                                                                                      layer_indices=currCF_gixs)
                            for p, g in zip(self.model.parameters(), currCF_reducedgrads):
                                p.grad = g

                            misc.sync_device(self.device)
                            logged_commtime = time.time() - comm_strt
                            logged_endtime = time.time() - begin_time

                            logged_tput = (self.world_size * self.train_bsz) / logged_endtime
                            logged_gain = currCF_compgain
                            logged_cf = curr_cf
                            logged_str = 'CURR_CF'

                        elif minCF_compgain >= self.epsilon > currCF_compgain:
                            # communicating gradients compressed to CF minCF
                            misc.sync_device(self.device)
                            comm_strt = time.time()
                            minCF_reducedgrads = self.comm_ops.compression_allgather(layer_values=minCF_gvals,
                                                                                     layer_indices=minCF_gixs)
                            for p, g in zip(self.model.parameters(), minCF_reducedgrads):
                                p.grad = g

                            misc.sync_device(self.device)
                            logged_commtime = time.time() - comm_strt
                            logged_endtime = time.time() - begin_time

                            logged_tput = (self.world_size * self.train_bsz) / logged_endtime
                            logged_gain = minCF_compgain
                            logged_cf = self.min_cf
                            logged_str = 'MIN_CF'

                        else:
                            # reduce uncompressed gradients
                            misc.sync_device(self.device)
                            comm_strt = time.time()
                            self.model = self.comm_ops.allreduce(self.model)
                            misc.sync_device(self.device)
                            logged_commtime = time.time() - comm_strt
                            logged_endtime = time.time() - begin_time

                            logged_tput = (self.world_size * self.train_bsz) / logged_endtime
                            logged_gain = 1.
                            logged_cf = 1.
                            logged_str = 'NO_COMPRESSION'

                else:
                    # run with minCF in the warmup period
                    misc.sync_device(self.device)
                    comm_strt = time.time()
                    minCF_reducedgrads = self.comm_ops.compression_allgather(layer_values=minCF_gvals,
                                                                             layer_indices=minCF_gixs)
                    for p, g in zip(self.model.parameters(), minCF_reducedgrads):
                        p.grad = g

                    misc.sync_device(self.device)
                    logged_commtime = time.time() - comm_strt
                    logged_endtime = time.time() - begin_time

                    logged_tput = (self.world_size * self.train_bsz) / logged_endtime
                    logged_gain = minCF_compgain
                    logged_cf = self.min_cf
                    logged_str = 'MIN_CF WARMUP'

                self.optimizer.step()
                self.optimizer.zero_grad()
                compression_throughput[logged_cf] = logged_tput * logged_gain
                logging.info(f'{logged_str} epoch {epoch} step {self.global_step} compute_time '
                             f'{compute_time} cf {self.min_cf} commtime {logged_commtime} compress_gain '
                             f'{logged_gain} comp_tput {compression_throughput[logged_cf]} tput {logged_tput}')

            if searchCF and self.global_step >= self.windowsize:
                curr_cf = (self.min_cf * self.cfstepsize) if curr_cf < self.max_cf else self.max_cf

            allcompression_tputs = list(compression_throughput.values())
            if len(allcompression_tputs) >= 2:
                allcompression_tputs.sort(reverse=True)
                max_compresstput = allcompression_tputs[0]
                second_max_compresstput = allcompression_tputs[1]
                threshold = abs((max_compresstput - second_max_compresstput) / second_max_compresstput)
                if threshold < self.omega:
                    for k, v in compression_throughput.items():
                        if v == max_compresstput:
                            cf1 = k
                        if v == second_max_compresstput:
                            cf2 = k

                    searchCF = False
                    if cf1 < cf2:
                        curr_cf = cf1
                    else:
                        curr_cf = cf2

                    logging.info(f'found threshold condition with CF {curr_cf} max_compresstput {max_compresstput} '
                                 f'and second_max_compresstput {second_max_compresstput}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='29501')
    parser.add_argument('--seed', type=int, default=1234, help='seed value for result replication')
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--log-dir', type=str, default='/', help='dir location when data is saved')
    parser.add_argument('--train-dir', type=str, default='/')
    parser.add_argument('--test-dir', type=str, default='/')
    parser.add_argument('--val-dir', type=str, default='/')
    parser.add_argument('--world-size', type=int, default=8, help='# procs to spawn across cluster')
    parser.add_argument('--rank', type=int, help='process rank', default=0)
    parser.add_argument('--epochs', type=int, help='# epochs to run', default=300)
    parser.add_argument('--compression', type=str, default='dgc')
    parser.add_argument('--compression-ratio', type=float, default=0.1)
    parser.add_argument('--compress-stages', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--val-bsz', type=int, default=128)
    parser.add_argument('--train-bsz', type=int, default=32)
    parser.add_argument('--trainacc-step', type=int, default=50)
    parser.add_argument('--do-compression', type=int, default=0, help='dont perform compression if this is 0.')
    parser.add_argument('--num-steps', type=int, default=35)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model-name', type=str, default='resnet101')

    parser.add_argument('--windowsize', type=int, default=25)
    parser.add_argument('--min-cf', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--omega', type=float, default=0.05)
    parser.add_argument('--epsilon', type=float, default=0.75)
    parser.add_argument('--compsteps', type=int, default=2000)
    parser.add_argument('--cfstepsize', type=float, default=0.5)
    parser.add_argument('--max-cf', type=float, default=0.0001)

    parser.add_argument('--dist-backend', type=str, default='gloo')
    parser.add_argument('--decentralized', type=int, default=1)
    parser.add_argument('--gpu-ix', type=int, default=0)
    parser.add_argument('--multi-gpu', type=int, default=0)
    parser.add_argument('--tid', type=int, default=1)
    parser.add_argument('--omp-threads', type=int, default=1)

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['OMP_NUM_THREADS'] = str(args.omp_threads)

    CompressionThroughputTraining(args).launch_training()