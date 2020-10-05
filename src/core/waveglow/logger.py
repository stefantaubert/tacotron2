from torch.utils.tensorboard import SummaryWriter


class WaveglowLogger(SummaryWriter):
  def __init__(self, logdir):
    super(WaveglowLogger, self).__init__(logdir)

  def log_training(self, reduced_loss, learning_rate, duration, iteration):
      self.add_scalar("training.loss", reduced_loss, iteration)
      self.add_scalar("learning.rate", learning_rate, iteration)
      self.add_scalar("duration", duration, iteration)

  def log_validation(self, reduced_loss, model, y, y_pred, iteration):
    self.add_scalar("validation.loss", reduced_loss, iteration)
