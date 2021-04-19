def pad(filename):
    # Load array
    original_array = np.load(filename)
    # Divide into pMHC and TCR
    pmhc = np.concatenate((original_array[original_array[:, 20] == 1], original_array[original_array[:, 21] == 1]))
    tcr = np.concatenate((original_array[original_array[:, 22] == 1], original_array[original_array[:, 23] == 1]))
    # Padding pMHC (only at the end)
    padding_size = (192 - pmhc.shape[0])
    end_pad = np.zeros((padding_size, pmhc.shape[1]))
    pmhc_padded = np.concatenate((pmhc, end_pad))
    # Padding TCR
    padding_size = (228 - tcr.shape[0]) / 2
    front_pad = np.zeros((math.floor(padding_size), tcr.shape[1]))
    end_pad = np.zeros((math.ceil(padding_size), tcr.shape[1]))
    tcr_padded = np.concatenate((front_pad, tcr, end_pad))
    # Concatanate pMHC and TCR
    array_padded = np.concatenate((pmhc_padded, tcr_padded))
    return array_padded

def load_data(filelist):
    padded_length = 192 + 228
    X = np.zeros(shape=(len(filelist), padded_length, 142))
    y = np.zeros(shape=len(filelist))
    for i in range(len(filelist)):
        filename = filelist[i]
        final_array = pad(filename)
        X[i] = final_array
        r = re.search(r'pos', filename)
        if r:
            y[i] = 1
        else:
            y[i] = 0

