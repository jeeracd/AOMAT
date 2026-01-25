from sklearn.metrics import root_mean_squared_error

def RMSE(rawData, candidateData):
    RMSE = root_mean_squared_error(rawData, candidateData, squared=False)
    return RMSE