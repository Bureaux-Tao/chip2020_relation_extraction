from keras.callbacks import ModelCheckpoint

from config import batch_size
from dataloader import load_data, data_generator
from evaluate import evaluate
from model import model, globalpointer_crossentropy
from path import weights_path, event_type, MODEL_TYPE, train_file_path, val_file_path
from plot import f1_plot, train_plot
from utils.backend import keras
from utils.adversarial import adversarial_training
from utils.optimizers import Adam, extend_with_exponential_moving_average

train_data = load_data(train_file_path)
valid_data = load_data(val_file_path)

# 构建模型
AdamEMA = extend_with_exponential_moving_average(Adam, name = 'AdamEMA')
optimizer = AdamEMA(lr = 1e-4)
model.compile(loss = globalpointer_crossentropy, optimizer = optimizer)
model.summary()

adversarial_training(model, 'Embedding-Token', 0.5)

f1_list = []
recall_list = []
precision_list = []

count_model_did_not_improve = 0


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    
    def __init__(self, patience = 1):
        super().__init__()
        self.best_val_f1 = 0.
        self.patience = patience
    
    def on_epoch_end(self, epoch, logs = None):
        global count_model_did_not_improve
        save_best_path = "{}/{}_{}_best.h5".format(weights_path, event_type, MODEL_TYPE)
        optimizer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_data, epoch)
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            count_model_did_not_improve = 0
            model.save_weights(save_best_path)
        else:
            count_model_did_not_improve += 1
            print("Early stop count " + str(count_model_did_not_improve) + "/" + str(self.patience))
            if count_model_did_not_improve >= self.patience:
                self.model.stop_training = True
                print("Epoch %05d: early stopping THR" % epoch)
        
        optimizer.reset_old_weights()
        print(
            'best f1: %.5f\n' % (self.best_val_f1)
        )


if __name__ == '__main__':
    save_all_path = ("{}/{}_{}_small".format(weights_path, event_type, MODEL_TYPE)) + "_ep{epoch:02d}.h5"
    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator(patience = 10)
    save_model = ModelCheckpoint(save_all_path, monitor = 'loss', verbose = 0, period = 1,
                                 save_weights_only = True, save_best_only = False)
    
    history = model.fit(
        train_generator.forfit(),
        steps_per_epoch = len(train_generator),
        epochs = 200,
        callbacks = [evaluator, save_model]
    )
    
    train_plot(history.history, history.epoch)
    
    data = {
        'epoch': range(1, len(f1_list) + 1),
        'f1': f1_list,
        'recall': recall_list,
        'precision': precision_list
    }
    
    f1_plot(data)