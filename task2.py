import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import random


def task_2_part_1(): 
    k = 0
    x = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])
    x = x.reshape(10,2)
    centroids = np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])
    while k < 3: 
        x_clr = []
        red = []
        green = []
        blue = []
        clr = ['red','green','blue']
        for i in range(len(x)): 
            d = 0
            d_old = np.inf
            for j in range(len(centroids)): 
                d = np.sqrt((x[i][0]-centroids[j][0])**2 + (x[i][1]- centroids[j][1])**2)
                print(d)
                if d < d_old: 
                    d_old = d
                    c_id = j
                    print(d_old,j)  
            x_clr.append(clr[c_id])
            if c_id == 0: 
                red.append(x[i])
            if c_id == 1:
                green.append(x[i])
            if c_id == 2:
                blue.append(x[i])
        plt.scatter(x[:,0], x[:,1],marker = "v",c = x_clr)
        plt.scatter(centroids[:,0],centroids[:,1],c = clr)
        plt.show(block = False)
        plt.savefig("Iter%d.jpg"%k)
        plt.pause(3)
        plt.close()
        centroids[0] = np.average(red,axis = 0)
        centroids[1] = np.average(green,axis = 0)
        centroids[2] = np.average(blue,axis = 0)
        k += 1

def subtask2(k):            
    img = cv2.imread('baboon.png')
    img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_NEAREST)
    image = img.reshape((img.shape[0] * img.shape[1], 3))
    # print(image.shape)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    k_clusters = k 
    im1 = []
    cltrs = image[np.random.choice(image.shape[0], k_clusters, replace=False)]
    print(cltrs)
    diff = np.inf
    while diff > 0.25:
        cp = {}
        avg_cp = {}
        order = []
        for i in image: 
            d = 0
            d_old = np.inf
            for n,j in enumerate(cltrs): 
                d = np.sqrt((i[0] - j[0])**2 + (i[1] - j[1])**2 + (i[2] - j[2])**2)
                if d < d_old: 
                    d_old = d
                    c_id = n
            if c_id not in cp:
                cp[c_id] = []
            order.append([i,c_id])
            cp[c_id].append(i)
        for k,v in cp.items(): 
            avg_cp[k] = np.average(v,axis = 0)
    #     print("*********",avg_cp)
        cltrs_new = []
        for i in range(len(cltrs)): 
            cltrs_new.append(avg_cp[i])
        diff = np.abs(np.average(np.array(cltrs_new) - np.array(cltrs)))
        print(diff)
        cltrs = cltrs_new
    tmp1 = cltrs
    tmp2 = cp 
    tmp3 = order
    for i in range(len(order)): 
        tmp3[i][0] = tmp1[tmp3[i][1]]
        im1.append(tmp3[i][0])
    image1 = np.array(im1).reshape((img.shape[0] , img.shape[1], 3))
    # image1 = cv2.cvtColor(np.uint8(image1), cv2.COLOR_BGR2RGB)
    cv2.imwrite("task2_baboon_%d.jpg"%k_clusters,np.uint8(image1))
    # image1 = cv2.cvtColor(np.uint8(image1), cv2.COLOR)
    # cv2.imwrite("%d_clusters1.jpg"%k_clusters,np.uint8(image1))
    # cv2.imshow('image1',image1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

for k in [3,5,10,20]:
    print("Current_Clusters = {}, Threshold = {}".format(k,0.25))
    subtask2(k)


# plt.hist(img.ravel(),256,[0,256]); 
# plt.show()



