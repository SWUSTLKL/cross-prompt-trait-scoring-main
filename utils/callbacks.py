import os

import scipy.signal
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.test_prompt_id = []
        self.attribute_name = []
        self.best_dev_epoch = []
        self.best_dev = []
        self.best_test = []

        
        os.makedirs(self.save_path)

    def append_loss(self, test_prompt_id, attribute_name, best_dev_epoch, best_dev, best_test):
        self.test_prompt_id.append(test_prompt_id)
        self.attribute_name.append(attribute_name)
        self.best_dev_epoch.append(best_dev_epoch)
        self.best_dev.append(best_dev)
        self.best_test.append(best_test)
        with open(os.path.join(self.save_path, "epoch_QWK_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(test_prompt_id))
            f.write("\n")
            f.write(str(attribute_name))
            f.write("\n")
            f.write(str(best_dev_epoch))
            f.write("\n")
            f.write(str(best_dev))
            f.write("\n")
            f.write(str(best_test))
            f.write("\n")
        # with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
        #     f.write(str(val_loss))
        #     f.write("\n")
        # self.loss_plot()

    # def loss_plot(self):
    #     iters = range(len(self.losses))
    #
    #     plt.figure()
    #     plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
    #     plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
    #     try:
    #         if len(self.losses) < 25:
    #             num = 5
    #         else:
    #             num = 15
    #
    #         plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
    #         plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
    #     except:
    #         pass
    #
    #     plt.grid(True)
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend(loc="upper right")
    #
    #     plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))
    #
    #     plt.cla()
    #     plt.close("all")
