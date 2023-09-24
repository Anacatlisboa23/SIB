class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray= None, features:List[str]=None, label: str= None)->None:
        """_summary_

        Args:
            X (np.ndarray): _description_
            y (np.ndarray, optional): _description_. Defaults to None.
            features (List[str], optional): _description_. Defaults to None.
            label (str, optional): _description_. Defaults to None.
        """
        if X is None:
            raise ValueError ('X must be defined')
        if y is not None and #continuar
        if X.shape [0] != y.shape[0]:
            raise ValueError ("The number of labels must be the same as the number of samples.")
        if features == None:
            features = [f"feat_{i}" for i in range (X.shape[0])]
        if label is not None:
            label = ['y'] 
        self.X=X
        self.y=y
        self.features=features
        self.label=label
        
    def shape (self):
        """shape of X.
        Returns the shape of X
        """
        return self.X.shape
    
    def has_label (self):
        if self.y is None:
            return False
        else:
            return True
        
    def get_classes (self):
        return np.unique(self.y)
    
    def get_mean(self):
        return np.nanmean(self.X, axis=0)
    
    def get_variance (self):
        return np.nanvar (self.X, axis=0)
    
    
    if __name__=="__main__":
        X = np.array([[1,2,3],[4,5,6]])
        data= Dataset (X=X,y=[1,0])
        print (data.get_mean())



        


