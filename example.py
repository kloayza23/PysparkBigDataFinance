def encode_variable(value_receive,number_range,incrementer):
    try:
        if(value_receive is None):
            return 0
        else:
            val_str=float(value_receive)
            val_comp=0.00
            for itera in range(number_range):
                if(val_str >= val_comp and val_str < (val_comp+incrementer)): 
                    return itera+1
                val_comp+=incrementer
            return 0
    except ValueError: 
        return 0
