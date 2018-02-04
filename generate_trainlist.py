def gene():
    trainlist = []
    for i in range(1,31):
        for j in range(1,23):
            trainlist.append([i*20,j*20,i*20+20,j*20,'right'])
            trainlist.append([i * 20, j * 20, i * 20 , j * 20+ 20, 'down'])
            trainlist.append([i * 20, j * 20, i * 20 -20, j * 20, 'left'])
            trainlist.append([i * 20, j * 20, i * 20 , j * 20-20, 'up'])
    return trainlist
if __name__ == '__main__':
    gene()