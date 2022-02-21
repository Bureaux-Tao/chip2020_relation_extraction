from keras.callbacks import ModelCheckpoint

from data_process import load_data, data_generator
from evaluate import evaluate
from path import *
from config import *
from plot import train_plot, f1_plot
from utils.adversarial import adversarial_training
from utils.backend import keras
from utils.optimizers import Adam, extend_with_exponential_moving_average, extend_with_piecewise_linear_lr
from casrel import train_model

train_data = load_data(train_file_path)
valid_data = load_data(val_file_path)

count_model_did_not_improve = 0

AdamEMA = extend_with_exponential_moving_average(Adam, name = 'AdamEMA')
optimizer = AdamEMA(lr = learning_rate)
train_model.compile(optimizer = optimizer)

adversarial_training(train_model, 'Embedding-Token', 0.5)

f1_list = []
recall_list = []
precision_list = []


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
            train_model.save_weights(save_best_path)
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
    save_all_path = ("{}/{}_{}_tiny".format(weights_path, event_type, MODEL_TYPE)) + "_ep{epoch:02d}.h5"
    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size)
    evaluator = Evaluator(patience = 10)
    save_model = ModelCheckpoint(save_all_path, monitor = 'loss', verbose = 0, period = 1,
                                 save_weights_only = True, save_best_only = False)
    
    # for i, item in enumerate(train_generator):
    #     print("\nbatch_token_ids shape:", item[0][0].shape)
    #     print("batch_segment_ids shape:", item[0][1].shape)
    #     print("batch_subject_labels shape:", item[0][2].shape)
    #     print("batch_subject_ids shape:", item[0][3].shape)
    #     print("batch_object_labels shape:", item[0][4].shape)
    #     if i == 4:
    #         break
    
    history = train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch = len(train_generator),
        validation_data = valid_generator.forfit(),
        validation_steps = len(valid_generator),
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
