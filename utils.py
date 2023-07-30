
def get_last_slice(arr, slice_num):
    slice_len = len(arr) // slice_num
    start_index = slice_len*(slice_num-1)
    end_index = slice_len*(slice_num)
    return arr[start_index:end_index]

def get_middle_slice(arr, slice_num):
    slice_len = len(arr) // slice_num
    start_index = slice_len*(slice_num//2)
    end_index = slice_len*((slice_num//2)+1)
    return arr[start_index:end_index]

def get_first_slice(arr, slice_num):
    slice_len = len(arr) // slice_num
    start_index = slice_len*(1)
    end_index = slice_len*(2)
    return arr[start_index:end_index]
