def make_flat(old_list):
    ''' Takes as argument a list of lists and flattens it (i.e. returns a 1D list with all the elements) '''
    new_list = []
    for elem in old_list :
        if type(elem)==list:
            elem_flat = make_flat(elem)
            new_list += elem_flat
        else :
            new_list.append(elem)
    return new_list
