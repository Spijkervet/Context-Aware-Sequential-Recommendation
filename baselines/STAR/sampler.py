from torch.utils.data import Dataset, DataLoader

class StarDataset(Dataset):
    ITEM_TRAIN = None
    HOUR_TRAIN = None
    WEEKDAY_TRAIN = None
    INTERVAL_TRAIN = None

    def __init__(self, ITEM_TRAIN,HOUR_TRAIN,WEEKDAY_TRAIN,INTERVAL_TRAIN):
        self.ITEM_TRAIN = ITEM_TRAIN
        self.HOUR_TRAIN = HOUR_TRAIN
        self.WEEKDAY_TRAIN = WEEKDAY_TRAIN
        self.INTERVAL_TRAIN = INTERVAL_TRAIN

    def __len__(self):
        return len(self.ITEM_TRAIN)

    def __getitem__(self, idx):
        user_cart = self.ITEM_TRAIN[idx]
        hour_cart = self.HOUR_TRAIN[idx]
        weekday_cart = self.WEEKDAY_TRAIN[idx]
        interval_cart = self.INTERVAL_TRAIN[idx]

        return {
                "user_cart":user_cart,
                "hour_cart":hour_cart,
                "weekday_cart":weekday_cart,
                "interval_cart":interval_cart
                }

""" 
for i in batch_inputs["user_cart"]:
				if len(i) > max_length:
					max_length = len(i)

			# pad the inputs to the longest boy
			for i, user_cart in enumerate(batch_inputs["user_cart"]):
				if len(cart) < max_length:
					difference = max_length - len(cart)

					hour_cart = batch_inputs["hour_cart"][i]
					weekday_cart = batch_inputs["weekday_cart"][i]
					interval_cart = batch_inputs["interval_cart"][i]

					batch_inputs["hour_cart"][i] = [0]*difference + list(hour_cart)
					batch_inputs["weekday_cart"][i] = [0]*difference + list(weekday_cart)
					batch_inputs["interval_cart"][i] = [0]*difference + list(interval_cart)

"""

class StarIdDataset(Dataset):
    ITEM_TRAIN = None
    HOUR_TRAIN = None
    WEEKDAY_TRAIN = None
    INTERVAL_TRAIN = None

    def __init__(self, ITEM_TRAIN):
        self.ITEM_TRAIN = ITEM_TRAIN

    def __len__(self):
        return len(self.ITEM_TRAIN)

    def __getitem__(self, idx):
        return idx

def get_dataloader(ITEM_TRAIN,HOUR_TRAIN,WEEKDAY_TRAIN,INTERVAL_TRAIN,batch_size=32):
    # target_dataset = StarDataset(ITEM_TRAIN,HOUR_TRAIN,WEEKDAY_TRAIN,INTERVAL_TRAIN)
    target_dataset = StarIdDataset(ITEM_TRAIN)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, num_workers=4)
    return target_dataloader

