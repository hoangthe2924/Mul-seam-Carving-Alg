import cv2
import numpy as np

def printSeam(I, j, backtrack):
    row, column,_ = I.shape
    for i in reversed(range(row)):
        # Danh dau vi tri cac phan tu se xoa
        I[i, j] = (0,0,255)  #thay thanh mau do de ve duong seam
        if (i != row-1):
            j = backtrack[i, j]
    return I

def computeEnergy(I):
    # B2: Tinh ma tran energy
    I = (I[:,:,0]+I[:,:,1]+I[:,:,2]).astype(np.float32)
    D_x = I[:-1, 1:].astype(np.float32) - I[:-1, :-1].astype(np.float32)
    D_y = I[1:, :-1].astype(np.float32) - I[:-1, :-1].astype(np.float32)
    D = np.sqrt(D_x**2 + D_y**2)
    return D

def computeMatrices(D):
    seamMatrix = D.copy()
    backtrack = np.zeros_like(D, dtype=np.int)
    parental_label = np.zeros_like(D, dtype=np.int)
    row, column = D.shape
    parental_label[0, :] = range(0, column)
    # Thuc hien tu goc trai buc anh
    for i in range(1, row):
        for j in range(0, column):
            if j == 0:
                index = np.argmin(seamMatrix[i - 1, j:j + 2])
                backtrack[i, j] = index + j
                min_energy = seamMatrix[i - 1, index + j]
                parental_label[i, j] = parental_label[i - 1, index]
            else:
                index = np.argmin(seamMatrix[i - 1, j - 1:j + 2])
                backtrack[i, j] = index + j - 1
                min_energy = seamMatrix[i - 1, index + j - 1]
                parental_label[i, j] = parental_label[i - 1, j - 1 + index]
            seamMatrix[i, j] += min_energy
    return seamMatrix, backtrack, parental_label


def findSeams(seamMatrix,parental_label, extant_seams):
    row, column = seamMatrix.shape
    rmv_list = []
    old_seam_index = parental_label[-1, 0]
    min_index = 0
    for i in range(1, column):
        if parental_label[-1, i] != old_seam_index:  #khac cha me thi them vao
            old_seam_index = parental_label[-1, i]
            rmv_list.append(min_index)
            min_index = i
            continue
        if seamMatrix[-1, i] < seamMatrix[-1, min_index]: #duong seam nho hon thi chon
            min_index = i

    rmv_list.append(min_index)
    rmv_list_num = len(rmv_list)
    if rmv_list_num >= extant_seams:
        rmv_list_num -= rmv_list_num - extant_seams
        rmv_list = rmv_list[0:rmv_list_num]
    return rmv_list

def seamsCarve(I, rmv_list, backtrack, extant_seams):
    row, column,_ = I.shape
    rmv_list_num = len(rmv_list)
    mask = np.ones((row, column), dtype=np.bool)
    for i in reversed(range(row)):
        for j in range(0, rmv_list_num):
            mask[i,rmv_list[j]] = False
        if(i!=row-1):
            rmv_list[:] = backtrack[i, rmv_list[:]]

    mask = np.stack([mask] * 3, axis=2)
    newI = I[mask].reshape(row , column - rmv_list_num,3)
    return newI, (extant_seams - rmv_list_num)


if __name__ == '__main__':
    # Step 1: Load input image

    image_dir = './example.jpg'
    I = cv2.imread(image_dir)

    num_seam = 100 # number of seams need to be carved
    extant_seams = num_seam # number of remaining seams need to be carved
    iter_count = 0
    while (extant_seams>0):
        iter_count += 1
        #Step 2: Compute all the matrices
        D = computeEnergy(I)

        # Step 3: Find the lowest-energy seam from left to right or top to bottom.
        seamMatrix, backtrack, parental_label = computeMatrices(D)

        # To show all the seams

        # row, column = D.shape
        # for j in range(column):
        #     Img = printSeam(I, j, backtrack)
        # cv2.imshow('Input', Img)
        # cv2.waitKey(0)

        rmv_list = findSeams(seamMatrix,parental_label,extant_seams)

        # Step 4: Carve the founding seams
        I, extant_seams = seamsCarve(I,rmv_list, backtrack, extant_seams)
        print('Iter %d: %d seams found' % (iter_count, len(rmv_list)))
        print('removed %d seam(s)' % (num_seam - extant_seams))

    # B5: Hien thi ket qua
    # cv2.imshow('Input', I)
    # cv2.waitKey(0)
    cv2.imwrite(image_dir.replace('.jpg','-carved.jpg'), I)
    

