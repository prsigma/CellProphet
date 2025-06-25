from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class Train_data(Dataset):
    """
    Custom Dataset class for training data that handles time series data with dropout masks.
    Supports train/validation split and sequential data extraction.
    """
    def __init__(self, expression_data,dropout_mask,size,flag):
        """
        Initialize the training dataset.
        
        Args:
            expression_data: The input time series data
            dropout_mask: Mask for dropout during training
            size: Tuple containing (input_length, prediction_length)
            flag: String indicating 'train' or 'val' for dataset split
        """
        self.expression_data = expression_data
        self.input_len = size[0]  # Length of input sequences
        self.pred_len = size[1]   # Length of prediction sequences
        self.mask = dropout_mask
        self.set_type = flag
        self.__read_data__()

    def __read_data__(self):
        """
        Split the data into training and validation sets based on the flag.
        Uses 80% for training and 20% for validation.
        """
        flag_dict = {'train':0,'val':1}
        
        # Calculate the split point (80% for training)
        num_train = int(len(self.expression_data) * 0.8)
        index = flag_dict[self.set_type]
        
        # Define borders for train/val split
        border1s = [0, num_train - self.input_len]  # Start indices
        border2s = [num_train, len(self.expression_data)]  # End indices

        border1 = border1s[index]
        border2 = border2s[index]
        
        # Extract the data portion based on the flag
        self.data_x = self.expression_data[border1:border2]
        self.data_y = self.expression_data[border1:border2]
        
        # Alternative: use full data 
        # self.data_x = self.expression_data
        # self.data_y = self.expression_data
        
    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            tuple: (input_sequence, target_sequence, target_mask)
        """
        # Define sequence boundaries
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        # Extract input and target sequences
        seq_x = self.data_x[s_begin:s_end]      # Input sequence
        seq_y = self.data_y[r_begin:r_end]      # Target sequence
        seq_y_mask = self.mask[r_begin:r_end]   # Mask for target sequence

        return seq_x, seq_y, seq_y_mask

    def __len__(self):
        """
        Return the total number of valid samples in the dataset.
        
        Returns:
            int: Number of samples that can be extracted
        """
        return len(self.data_x) - self.input_len - self.pred_len + 1

class Test_data(Dataset):
    """
    Custom Dataset class for test data that handles sequential input extraction
    without requiring target sequences or masks.
    """
    def __init__(self, expression_data,size):
        """
        Initialize the test dataset.
        
        Args:
            expression_data: The input time series data for testing
            size: Tuple containing input length (only first element is used)
        """
        self.expression_data = expression_data
        self.input_len = size[0]  # Length of input sequences
        
    def __getitem__(self, index):
        """
        Get a single input sequence from the test dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            tensor: Input sequence for prediction
        """
        # Calculate sequence boundaries
        s_begin = index * self.input_len
        s_end = s_begin + self.input_len
        
        # Handle case where sequence extends beyond data length
        if s_end > len(self.expression_data):
            # Use the last available sequence of required length
            seq_x = self.expression_data[-self.input_len:]
        else:
            # Extract normal sequence
            seq_x = self.expression_data[s_begin:s_end]

        return seq_x

    def __len__(self):
        """
        Return the total number of samples that can be extracted from test data.
        
        Returns:
            int: Number of non-overlapping sequences plus one for remainder
        """
        return len(self.expression_data) // self.input_len + 1
