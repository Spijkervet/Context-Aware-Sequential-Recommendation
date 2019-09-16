
class Data():

    def __init__(self):
        pass


    def to_int(self):
        product_ids = sorted(set(self.df['asin']))
        d = {}
        i = 0
        for pi in product_ids:
            if pi not in d:
                d[pi] = i
                self.df.loc[self.df['asin'] == pi, 'product_id'] = i
                i += 1
                
        self.id2int = d
        return self.id2int