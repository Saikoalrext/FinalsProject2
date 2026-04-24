import math

def cosine(v1, v2):
    dot= 0
    norm1= 0
    norm2= 0

    for t in v1:
        norm1+= v1[t]** 2
        if t in v2:
            dot+= v1[t]* v2[t]

    for t in v2:
        norm2+= v2[t]** 2

    if norm1== 0 or norm2== 0:
        return 0

    return dot/ (math.sqrt(norm1)* math.sqrt(norm2))