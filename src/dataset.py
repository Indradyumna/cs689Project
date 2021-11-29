from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import pickle
import os



class GraphDataset(object):
  """
  """
  def __init__(self,av):
    self.av = av
    self.load_graph()
    self.create_data_splits()

  def load_graph(self): 
    fname = "Datasets/" + self.av.DATASET_NAME + "_689_data1.pkl"
    print (fname)
    if os.path.isfile(fname):
        self.dataset = pickle.load(open(fname,"rb"))
    else:  
        self.dataset = TUDataset(root=".", name=self.av.DATASET_NAME)

    with open(fname,"wb") as f:
        pickle.dump(self.dataset,f)

  def create_data_splits(self):
    self.num_total_data = len(self.dataset)
    tr_idx_end = int(self.num_total_data * 0.7)
    val_idx_end = int(self.num_total_data * 0.8)
    self.num_val_data = val_idx_end - tr_idx_end
    self.num_test_data = self.num_total_data - val_idx_end 
    self.loader_train = DataLoader(self.dataset[:tr_idx_end], batch_size=self.av.BATCH_SIZE, shuffle=True)
    self.loader_valid = DataLoader(self.dataset[tr_idx_end:val_idx_end], batch_size=self.num_val_data, shuffle=True)
    self.loader_test = DataLoader(self.dataset[val_idx_end:], batch_size=self.num_test_data, shuffle=False)

    
