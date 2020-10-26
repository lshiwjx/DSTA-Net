from train_val_test import train_val_model
import shutil
import inspect


def train_val_choose(args, block):
    if args.train == 'classify':
        train_net = train_val_model.train_classifier
        val_net = train_val_model.val_classifier
    else:
        raise ValueError("args of train val is not right")

    shutil.copy2(inspect.getfile(train_net), args.model_saved_name)
    shutil.copy2(__file__, args.model_saved_name)

    return train_net, val_net


if __name__ == '__main__':
    train_net = train_val_model.train_classifier
    print(inspect.getfile(train_net))
