import numpy as np
x=np.array([  1060, 126481,   2127,    417,  63597, 248315,  13326,   7063,   1589,
        248355,   6212,  71349,  72607,    553,   5620,  72811,  54505,  27499,
        248171,    735,   5620,  72811,  54505,  27499, 248171,    735,   5620,
         72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,  27499,
        248171,    735,   5620,  72811,  54505,  27499, 248171,    735,   5620,
         72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,  27499,
        248171,    735,   5620,  72811,  54505,  27499, 248171,    735,   5620,
         72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,  27499,
        248171,    735,   5620,  72811,  54505,  27499, 248171,    735,   5620,
         72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,  27499,
        248171,    735,   5620,  72811,  54505,  27499, 248171,    735,   5620,
         72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,  27499,
        248171,    735,   5620,  72811,  54505,  27499, 248171,    735,   5620,
         72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,  27499,
        248171,    735,   5620,  72811,  54505,  27499, 248171,    735,   5620,
         72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,  27499,
        248171, 248075,      2])

# def find_most_repeating_pattern(tensor):
#     tensor_length = len(tensor)
#     most_repeats = 0
#     most_repeating_pattern = None
    
#     for pattern_length in range(1, tensor_length // 2 + 1):  # Only need to check up to half the length of the tensor
#         is_repeating = True
#         repeats = 0
#         for i in range(0, tensor_length - pattern_length, pattern_length):
#             # Compare each slice with the next one of the same length
#             if np.array_equal(tensor[i:i+pattern_length], tensor[i+pattern_length:i+2*pattern_length]):
#                 if i + 2 * pattern_length <= tensor_length:
#                     repeats += 1
#             else:
#                 is_repeating = False
#                 break
#         if is_repeating and repeats > most_repeats:  # Check if this pattern has the most repeats so far
#             most_repeats = repeats
#             most_repeating_pattern = tensor[:pattern_length]
    
#     return most_repeating_pattern, most_repeats + 1  # Include the initial occurrence in the count

def find_most_repeating_pattern(tensor):
    tensor_length = len(tensor)
    max_pattern_length = 0
    max_repeats = 0
    most_repeating_pattern = None
    
    # Search for all possible patterns by their starting position and length
    for start in range(tensor_length):
        for pattern_length in range(1, tensor_length - start + 1):
            repeats = 1
            pattern = tensor[start:start+pattern_length]
            # Look ahead to see if the pattern repeats consecutively
            for next_start in range(start + pattern_length, tensor_length - pattern_length + 1, pattern_length):
                next_pattern = tensor[next_start:next_start+pattern_length]
                if np.array_equal(pattern, next_pattern):
                    repeats += 1
                else:
                    break  # Break if the pattern does not continue consecutively
            
            # Update the most repeating pattern if this pattern repeats more consecutively
            if repeats > max_repeats or (repeats == max_repeats and pattern_length > max_pattern_length):
                max_repeats = repeats
                max_pattern_length = pattern_length
                most_repeating_pattern = pattern
    
    return most_repeating_pattern, max_repeats

print(find_most_repeating_pattern(x))

def detect_repeating_pattern_entries(tensor):
    # The identified repeating pattern
    pattern = [5620, 72811, 54505, 27499, 248171, 735]
    pattern_length = len(pattern)
    tensor_length = len(tensor)
    
    # Initialize a boolean array to mark entries part of the repeating pattern
    is_part_of_pattern = np.zeros(tensor_length, dtype=bool)
    for i in range(tensor_length - pattern_length + 1):
        # Check if the current window matches the repeating pattern
        if np.array_equal(tensor[i:i+pattern_length], pattern):
            is_part_of_pattern[i:i+pattern_length] = True
    return is_part_of_pattern


print(detect_repeating_pattern_entries(x))
print(x[14:])

import torch
x=torch.IntTensor([ 77636,  65164,   1060, 126481,   2127,    417,  63597, 248315,  13326,
          7063,   1589, 248355,   6212,  71349,  72607,    553,   5620,  72811,
         54505,  27499, 248171,    553,   5620,  72811,  54505,  27499, 248171,
           735,   8582,   2127,  19499, 248260, 164671,  71349,  77051,    553,
          5620,  72811,  54505,  27499, 248171,    553,   5620,  72811,  54505,
         27499, 248171,    553,   5620,  72811,  54505,  27499, 248171,    553,
          5620,  72811,  54505,  27499, 248171,    735,   8582,   2127,  19499,
        248260, 248075,  77636,  41833,   1060, 126481,   2127,    417,  63597,
        248315,  13326,   7063,   1589, 248355,   6212,  71349,  72607,    553,
          5620,  72811,  54505,  27499, 248171,    553,   5620,  72811,  54505,
         27499, 248171,    553,   5620,  72811,  54505,  27499, 248171,    735,
          5620,  72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,
         27499, 248171,    735,   5620,  72811,  54505,  27499, 248171,    735,
          5620,  72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,
         27499, 248171,    735,   5620,  72811,  54505,  27499, 248171,    735,
          5620,  72811,  54505,  27499, 248171,    735,   5620,  72811,  54505,
         27499, 248171, 248075,      2])[:-4]
x=x[None,:]

y = torch.IntTensor([77636, 65164, 1060, 126481, 2127, 417, 63597, 248315, 13326, 7063, 1589, 248355, 6212, 71349, 72607, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 735, 8582, 2127, 19499, 248260, 164671, 71349, 77051, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 735, 8582, 2127, 19499, 248260, 248075, 77636, 41833, 1060, 126481, 2127, 417, 63597, 248315, 13326, 7063, 1589, 248355, 6212, 71349, 72607, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 553, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 735, 5620, 72811, 54505, 27499, 248171, 248075, 2])[:-4][None, :]

assert (y==x).all()