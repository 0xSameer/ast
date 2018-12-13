# coding: utf-8

"""
Train models
"""

from nn import *

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=program_descrp)
    parser.add_argument('-m','--cfg_path', help='path for model config',
                        required=True)
    parser.add_argument('-e','--epochs', help='num epochs',
                        required=True)

    args = vars(parser.parse_args())

    cfg_path = args['cfg_path']

    epochs = int(args['epochs'])
    print("number of epochs={0:d}".format(epochs))
    
    """
    Create the model and load previously stored parameters
    """
    nn = NN(cfg_path)
    train_key = nn.cfg.train["train_set"]
    dev_key = nn.cfg.train["dev_set"]
    iters_save = nn.cfg.train['iters_save']
    model_fil = nn.model_fname
    """
    Load references for evaluation
    """
    refs_path = os.path.join(nn.cfg.train["data"]["refs_path"],
                             dev_key)
    metrics = Eval(refs_path, nn.cfg.train["data"]["n_evals"])

    curr_epoch = nn.max_epoch + 1
    max_epoch = curr_epoch+epochs
    for epoch in range(curr_epoch, max_epoch):
        print("Epoch: {0:d}".format(curr_epoch))

        # Train model
        epoch_loss = nn.train_epoch(train_key)
        print("Loss = {0:.4f}".format(epoch_loss))
        with open(nn.train_log, mode='a') as train_log:
            # log train loss
            train_log.write("{0:d}, {1:.4f}\n".format(epoch, epoch_loss))
        # Evaluate model
        preds = nn.predict(dev_key)
        hyps = nn.data_loader.get_hyps(preds)
        bleu = metrics.calc_bleu(hyps) * 100

        with open(nn.dev_log, mode='a') as dev_log:
            # log dev bleu
            dev_log.write("{0:d}, {1:.2f}\n".format(epoch, bleu))
        print("BLEU = {0:.2f}".format(bleu))

        # Save model
        if ((epoch % iters_save == 0) or (epoch == max_epoch-1)):
            print("Saving model")
            serializers.save_npz(model_fil.replace(".model", "_{0:d}.model".format(epoch)), nn.model)
            print("Finished saving model")
# -----------------------------------------------------------------------------
