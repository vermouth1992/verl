from verl.single_controller.base import Worker

from verl import DataProto

from abc import abstractmethod

# used for LLM, VLM, actor/reference/generative RM, etc,.
class LLMWorker(Worker):

    @abstractmethod
    def _reduce_output(self, model_output):
        pass

    @abstractmethod
    def infer_batch(self, data: DataProto) -> DataProto:
        pass
    
    @abstractmethod
    def set_loss_function(self, loss_fn):
        """
        This function enables to modify loss_function at runtime
        """
        pass
        
    @abstractmethod
    def train_batch(self, data: DataProto) -> DataProto:
        # implement training one batch here
        pass
        
    @abstractmethod
    def save_checkpoint(self):
        pass
    
    @abstractmethod
    def load_checkpoint(self):
        pass