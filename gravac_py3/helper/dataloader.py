import random
import numpy
from PIL import Image

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from gravac_py3.helper import misc
import gravac_py3.helper.ptb_dataloader as ptb_reader


class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartioner(object):
    def __init__(self, data, world_size):
        self.data = data
        self.partitions = []
        # partition data equally among the trainers
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)
        partitions = [1 / (world_size) for _ in range(0, world_size)]
        for part in partitions:
            part_len = int(part * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def traindata_CIFAR10(train_dir, world_size, trainer_rank, train_bsz, seed, num_workers=1):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    trainer_index = trainer_rank

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    trainset = torchvision.datasets.CIFAR10(root=train_dir + 'data', train=True,
                                            download=True, transform=transform)
    partition = DataPartioner(trainset, world_size)
    print(f"going to use ix {trainer_index} for partition")
    partition = partition.use(trainer_index)
    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                                worker_init_fn=misc.set_seed(seed), generator=g,
                                                num_workers=num_workers)

    return trainloader, len(trainset)


def testdata_CIFAR10(test_dir, test_bsz, seed):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR10(root=test_dir + 'data', train=False,
                                           download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=False, generator=g, num_workers=1)
    return testloader


def traindata_CIFAR100(train_dir, world_size, trainer_rank, train_bsz, seed, num_workers=1):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    trainer_index = trainer_rank

    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32,4), normalize])
    trainset = torchvision.datasets.CIFAR100(root=train_dir, train=True,
                                             download=True, transform=transform)
    partition = DataPartioner(trainset, world_size)
    print(f"going to use ix {trainer_index} for partition")
    partition = partition.use(trainer_index)
    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g,
                                              num_workers=num_workers)
    return trainloader, len(trainset)


def testdata_CIFAR100(test_dir, test_bsz, seed):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR100(root=test_dir, train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=False, generator=g, num_workers=1)
    return testloader


def traindata_ImageNet(train_dir, world_size, trainer_rank, train_bsz, seed, num_workers=1):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    trainer_index = trainer_rank

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    size = (224, 256)
    train_dataset = datasets.ImageFolder(train_dir, transforms.Compose([transforms.RandomResizedCrop(size[0]),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(), normalize]))
    partition = DataPartioner(train_dataset, world_size)
    print(f"going to use ix {trainer_index} for partition")
    partition = partition.use(trainer_index)
    train_loader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                                 worker_init_fn=misc.set_seed(seed), generator=g,
                                                 num_workers=num_workers)
    return train_loader, len(train_dataset)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            sqrWidth = numpy.ceil(numpy.sqrt(img.size[0]*img.size[1])).astype(int)
            return img.resize((sqrWidth, sqrWidth))


def testdata_ImageNet(test_dir, test_bsz, seed, num_workers=1):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    size = (224, 256)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(size[1]), transforms.CenterCrop(size[0]),
                                    transforms.ToTensor(), normalize])
    test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(test_dir, transform=transform, loader=pil_loader),
                                                batch_size=test_bsz, shuffle=False, generator=g,
                                                num_workers=num_workers)
    return test_loader


def validationdata_ImageNet(val_dir, val_bsz, seed, num_workers=1):
    misc.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    size = (224, 256)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(size[1]), transforms.CenterCrop(size[0]),
                                    transforms.ToTensor(), normalize])
    validation_loader = torch.utils.data.DataLoader(datasets.ImageFolder(val_dir, transform=transform),
                                                    batch_size=val_bsz, shuffle=False, generator=g,
                                                    num_workers=num_workers)
    return validation_loader


def traindata_PTB(data_dir, world_size, rank, batch_size=20, num_workers=1, num_steps=35, seed=1234):
    misc.set_seed(seed=seed)
    g = torch.Generator()
    g.manual_seed(seed)

    raw_data = ptb_reader.ptb_raw_data(data_path=data_dir)
    train_data,_,_,word_to_id,_ = raw_data
    vocab_size = len(word_to_id)
    train_set = ptb_reader.PTBTrainData(train_data, batch_size, num_steps)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_sampler.set_epoch(0)
    trainloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, num_workers=num_workers,
                                              pin_memory=True, sampler=train_sampler, drop_last=True,
                                              worker_init_fn=misc.set_seed(seed), generator=g)
    return trainloader, vocab_size, len(train_set)


def testdata_PTB(data_dir, batch_size=20, num_workers=1, num_steps=35, seed=1234):
    misc.set_seed(seed=seed)
    g = torch.Generator()
    g.manual_seed(seed)
    raw_data = ptb_reader.ptb_raw_data(data_path=data_dir)
    _, valid_data, _, word_to_id, _ = raw_data
    vocab_size = len(word_to_id)
    test_set = ptb_reader.PTBTestData(valid_data, batch_size, num_steps)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True, generator=g,
                                             drop_last=True, worker_init_fn=misc.set_seed(seed))
    return testloader, vocab_size


class TrainingData(object):
    def __init__(self, args, trainer_batch_size, trainer_rank):
        self.args = args
        self.trainer_batch_size = trainer_batch_size
        self.trainer_rank = trainer_rank

    def train_data(self):
        train_loader, vocab_size, dataset_size = None, None, None
        if self.args.dataset == 'cifar10':
            train_loader, dataset_size = traindata_CIFAR10(train_dir=self.args.train_dir, world_size=self.args.world_size,
                                                           trainer_rank=self.trainer_rank, train_bsz=self.trainer_batch_size,
                                                           seed=self.args.seed, num_workers=1)
        elif self.args.dataset == 'cifar100':
            train_loader, dataset_size = traindata_CIFAR100(train_dir=self.args.train_dir, world_size=self.args.world_size,
                                                            trainer_rank=self.trainer_rank, train_bsz=self.trainer_batch_size,
                                                            seed=self.args.seed)
        elif self.args.dataset == 'imagenet':
            train_loader, dataset_size = traindata_ImageNet(train_dir=self.args.train_dir, world_size=self.args.world_size,
                                                            trainer_rank=self.trainer_rank, train_bsz=self.trainer_batch_size,
                                                            seed=self.args.seed, num_workers=4)
        elif self.args.dataset == 'ptb':
            train_loader, vocab_size, dataset_size = traindata_PTB(data_dir=self.args.log_dir, rank=self.trainer_rank,
                                                                   world_size=int(self.args.world_size),
                                                                   num_workers=1, num_steps=self.args.num_steps,
                                                                   seed=self.args.seed, batch_size=self.trainer_batch_size)

        return train_loader, vocab_size, dataset_size


class TestData(object):
    def __init__(self, args):
        self.args = args

    def test_data(self):
        testloader, vocab_size = None, None
        if self.args.dataset == 'cifar10':
            testloader = testdata_CIFAR10(test_dir=self.args.test_dir, test_bsz=self.args.test_bsz, seed=self.args.seed)

        elif self.args.dataset == 'cifar100':
            testloader = testdata_CIFAR100(test_dir=self.args.test_dir, test_bsz=self.args.test_bsz, seed=self.args.seed)

        elif self.args.dataset == 'imagenet':
            testloader = testdata_ImageNet(test_dir=self.args.test_dir, test_bsz=self.args.test_bsz, seed=self.args.seed)

        elif self.args.dataset == 'ptb':
            testloader, vocab_size = testdata_PTB(data_dir=self.args.log_dir, batch_size=self.args.test_bsz,
                                                  num_workers=1, num_steps=self.args.num_steps, seed=self.args.seed)

        return testloader, vocab_size