import os
from package import *
from package.args.main_args import parse_config
args = parse_config()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from package.models.gcnclf import *
from package.dataset.data_mnist import *
from sklearn.neighbors import NearestNeighbors as NN
import sys
import time


DEBUG = 0
IS_WIN32 = sys.platform == 'win32'


def _eval(data, model, batch_size=256):
    model.eval()
    hits = 0
    total = 0
    with torch.no_grad():
        for xs, label in data.traverse(batch_size=batch_size):
            # print(label)
            out = model(xs.cuda())
            hits += float((torch.argmax(out, dim=-1) == label.cuda()).sum().cpu())
            total += len(label)
    model.train()
    return hits / total


def _test_and_save(epochs, data_test, model, logger, args, loss_sum):
    if not hasattr(_test_and_save, 'best_acc'):
        _test_and_save.best_acc = 0
    start_cpu_t = time.time()
    pre = _eval(model=model, data=data_test, batch_size=args.batch_size*3)

    logger.info("Precision: {}, bestPrecsion: {}".format(
        pre, max(pre, _test_and_save.best_acc)) +
                "\n" + 'epochs: {},  loss: {},  (eval cpu time: {}s)'.
                format(epochs, [np.mean(loss) for loss in loss_sum], time.time() - start_cpu_t))

    if pre >= _test_and_save.best_acc - 0.5 and epochs != 0:
        _test_and_save.best_acc = pre
        d = {'model': model.state_dict(),
                    'epochs': epochs,
                    'args': args}
        torch.save(d, save_fn(args.save_dir, epochs, pre, 0))
    torch.cuda.empty_cache()


def save_fn(save_dir, it, pre=0, mAP=0):
    return join(mkdir(join(save_dir, 'models')), 'Iter__{}__{}_{}.pkl'.format(it, int(pre * 1000), int(mAP * 1000)))


def _try_load(args, logger, model):
    if args.start_from is None:
        # try to find the latest checkpoint
        files = os.listdir(mkdir(join(mkdir(args.save_dir), 'models')))
        if len(files) == 0:
            logger.info("Cannot find any checkpoint. Start new training.")
            return 0
        latest = max(files, key=lambda name: int(name.split('\\')[-1].split('/')[-1].split('.')[0].split('__')[1]))
        checkpoint = join(args.save_dir, 'models', latest)
    else:
        try: checkpoint = save_fn(args.save_dir, str(int(args.start_from)))
        except: checkpoint = args.start_from
    logger.info("Load model from {}".format(checkpoint))
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    return ckpt['epochs']


def _init_dataset(args):
    logger = make_logger(join(mkdir(args.save_dir), curr_time_str() + '.log'))
    data_train = Mnist(train=True)
    data_test = Mnist(train=False)
    dataloader_train = DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=True)
    return data_train, dataloader_train, data_test, logger


def _train(args, dataloader_train, data_test, logger):
    model = GCNCLF(layers=[28, 64, 64, 64, 'max', 64, 10], logger=logger, lr=args.lr, method='ful')
    model = GCNCLF(layers=[28, 64, 64, -8, 128, 128, 'max', 32, 10], logger=logger, lr=args.lr, method='ful')
    model = GCNCLF(layers=[28, 64, 128, 'max', 128, 10], logger=logger, lr=args.lr, method='dg2')
    model = GCNCLF(layers=[28, 64, 128, -1, 128, 10], logger=logger, lr=args.lr, method='dg2', cat=False)
    model = GCNCLF(layers=[28, 64, 128, -4, 256, 'max', 128, 10], logger=logger, lr=args.lr, method='dg2')
    print(model)
    model.cuda()
    epochs = _try_load(args, logger, model)
    logger.info(str(args))
    args.epochs += epochs
    model.train()
    loss_sum = [[0] for _ in range(model.loss_num)]
    _test_and_save(epochs=epochs, data_test=data_test,
                   model=model, logger=logger, args=args, loss_sum=loss_sum)
    steps = 0
    def prt(loss_sum):
        logger.info('epochs: {},  loss: {},  steps: {}'.
                    format(epochs, [np.mean(loss) for loss in loss_sum], steps))
        return loss_sum

    while True:
        for _, batch in enumerate(dataloader_train):
            steps += 1
            for i in range(len(batch)):
                batch[i] = batch[i].cuda()
            img, label = batch
            loss = model.optimize_params(img, label)
            for i in range(len(loss_sum)):
                loss_sum[i].append(float(loss[i]))

            if steps % args.save_every == 0:
                _test_and_save(epochs=epochs, data_test=data_test,
                               model=model, logger=logger, args=args, loss_sum=loss_sum)
            if steps % args.print_every == 0:
                loss_sum = prt(loss_sum)
                loss_sum = [[loss[-1]] for loss in loss_sum]

        epochs += 1
        if epochs >= args.epochs: break


def train(args):
    data_train, dataloader_train, data_test, logger = _init_dataset(args=args)
    _train(args=args, data_test=data_test, dataloader_train=dataloader_train, logger=logger)


if __name__ == '__main__':
    train(args)


