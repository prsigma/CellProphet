from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class MTGRN_data(Dataset):
    def __init__(self, expression_data,dropout_mask,size,flag):
        self.expression_data = expression_data
        self.input_len = size[0]
        self.pred_len = size[1]
        self.mask = dropout_mask
        self.set_type = flag
        self.__read_data__()

    def __read_data__(self):
        flag_dict = {'train':0,'val':1}
        
        num_train = int(len(self.expression_data) * 0.8)
        index = flag_dict[self.set_type]
        
        border1s = [0, num_train - self.input_len]
        border2s = [num_train, len(self.expression_data)]

        border1 = border1s[index]
        border2 = border2s[index]
        
        self.data_x = self.expression_data[border1:border2]
        self.data_y = self.expression_data[border1:border2]
        
        # self.data_x = self.expression_data
        # self.data_y = self.expression_data
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_y_mask = self.mask[r_begin:r_end]

        return seq_x, seq_y, seq_y_mask

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1

class MTGRN_test_data(Dataset):
    def __init__(self, expression_data,size):
        self.expression_data = expression_data
        self.input_len = size[0]
        
    def __getitem__(self, index):
        s_begin = index * self.input_len
        s_end = s_begin + self.input_len
        
        if s_end > len(self.expression_data):
            seq_x = self.expression_data[-self.input_len:]
        else:
            seq_x = self.expression_data[s_begin:s_end]

        return seq_x

    def __len__(self):
        return len(self.expression_data) // self.input_len + 1
