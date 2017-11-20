def chunks(l, n):
    chunks = []
    for i in range(0, len(l), n):
        chunks.append(l[i:i + n])
    return chunks