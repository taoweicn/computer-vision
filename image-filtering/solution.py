import cv2
import numpy as np
import matplotlib.pyplot as plt


def bilateralFilter(img, radius, sigmaColor, sigmaSpace):
    B, G, R = cv2.split(img)
    B_tran, G_tran, R_tran = cv2.split(img)
    img_height = len(B)
    img_width = len(B[0])
    # 计算灰度值模板系数表
    color_coeff = -0.5 / (sigmaColor * sigmaColor)
    weight_color = []  # 存放颜色差值的平方
    for i in range(256):
        weight_color.append(np.exp(i * i * color_coeff))
    # 计算空间模板
    space_coeff = -0.5 / (sigmaSpace * sigmaSpace)
    weight_space = []  # 存放模板系数
    weight_space_row = []  # 存放模板 x轴 位置
    weight_space_col = []  # 存放模板 y轴 位置
    maxk = 0
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            r_square = i * i + j * j
            r = np.sqrt(r_square)
            weight_space.append(np.exp(r_square * space_coeff))
            weight_space_row.append(i)
            weight_space_col.append(j)
            maxk = maxk + 1
    # 进行滤波
    for row in range(img_height):
        for col in range(img_width):
            value_R = 0
            value_G = 0
            value_B = 0
            weight_R = 0
            weight_G = 0
            weight_B = 0
            for i in range(maxk):
                m = row + weight_space_row[i]
                n = col + weight_space_col[i]
                if m < 0 or n < 0 or m >= img_height or n >= img_width:
                    val_R = 0
                    val_G = 0
                    val_B = 0
                else:
                    val_R = R[m][n]
                    val_G = G[m][n]
                    val_B = B[m][n]
                w_R = np.float32(weight_space[i]) * np.float32(weight_color[np.abs(val_R - R[row][col])])
                w_G = np.float32(weight_space[i]) * np.float32(weight_color[np.abs(val_G - G[row][col])])
                w_B = np.float32(weight_space[i]) * np.float32(weight_color[np.abs(val_B - B[row][col])])
                value_R += val_R * w_R
                value_G += val_G * w_G
                value_B += val_B * w_B
                weight_R += w_R
                weight_G += w_G
                weight_B += w_B
            R_tran[row][col] = np.uint8(value_R / weight_R)
            G_tran[row][col] = np.uint8(value_G / weight_G)
            B_tran[row][col] = np.uint8(value_B / weight_B)
    return cv2.merge([B_tran, G_tran, R_tran])


def plot(images, title):
    fig = plt.figure()
    fig.suptitle(title)
    num = len(images)
    for index in range(num):
        ax = fig.add_subplot(1, num, index + 1)
        ax.set_title('img' + str(index + 1))
        ax.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
        ax.axis('off')


def main():
    image = cv2.imread('./images/img1.jpg')

    plt.title('Original')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    window_size = [(3, 3), (5, 5), (7, 7)]

    images = [cv2.blur(image, size, 0) for size in window_size]
    plot(images, 'blur')

    images = [cv2.GaussianBlur(image, size, 0) for size in window_size]
    plot(images, 'GaussianBlur')

    params = [(5, 21, 21), (7, 31, 31), (9, 41, 41)]
    images = [cv2.bilateralFilter(image, *param) for param in params]
    plot(images, 'bilateralFilter')

    images = [bilateralFilter(image, *param) for param in params]
    plot(images, 'bilateralFilter')

    plt.show()


if __name__ == '__main__':
    main()
