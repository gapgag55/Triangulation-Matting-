import numpy as np
import cv2


def findMatting(compA, compB, backA, backB):

    compAr, compAg, compAb = compA[:, :, 0], compA[:, :, 1], compA[:, :, 2]
    compBr, compBg, compBb = compB[:, :, 0], compB[:, :, 1], compB[:, :, 2]
    backAr, backAg, backAb = backA[:, :, 0], backA[:, :, 1], backA[:, :, 2]
    backBr, backBg, backBb = backB[:, :, 0], backB[:, :, 1], backB[:, :, 2]

    # Size and Color
    img_shape = compA.shape
    fg = np.zeros(compA.shape)

    # Only Size
    alpha = np.zeros(img_shape[:2])

    IndentifiedMatrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            back = np.array([
                [backAr[i, j]], [backAg[i, j]], [backAb[i, j]],
                [backBr[i, j]], [backBg[i, j]], [backBb[i, j]]
            ])
            deltaCompAndBg = np.array([
                [compAr[i, j] - backAr[i, j]], [compAg[i, j] - backAg[i, j]], [compAb[i, j] - backAb[i, j]],
                [compBr[i, j] - backBr[i, j]], [compBg[i, j] - backBg[i, j]], [compBb[i, j] - backBb[i, j]]
            ])

            # Combine background Array with Indentified Matrix.
            A = np.hstack((IndentifiedMatrix, -1 * back))

            # Clip and Get only Image
            # Solve equation np.linalg.pinv(A) --> get [R1, G1, B1, R2, G2, B2]
            # Product with deltaCompAndBg
            # Then clip only visible image or value
            clip = np.clip(np.dot(np.linalg.pinv(A), deltaCompAndBg), 0.0, 1.0)

            # Extracting Foregound
            fg[i, j] = np.array([clip[0][0], clip[1][0], clip[2][0]])

            # Get alpha
            alpha[i, j] = clip[3]

    return fg, alpha

    # Extracting Foregound
    fg[i, j] = np.array([clip[0][0], clip[1][0], clip[2][0]])

    # Get alpha
    alpha[i, j] = clip[3]

    return fg, alpha

def modifyBackgroundWithAlpha(bg, alpha):

    newBg = np.zeros(bg.shape)

    for height in range(bg.shape[0]):
        for width in range(bg.shape[1]):
            newBg[height][width] = bg[height][width] * (1.0 - alpha[height][width])

    return newBg;

def work(b1, b2, c1, c2, newBg):

    # NOTE: We use division with 255.0 because our calculation is out of bound 255 such as 253, 256
    # We use ratio technique by dividing with 255.0 to calculate matrix
    backA = cv2.imread(b1)/255.0
    backB = cv2.imread(b2)/255.0
    compA = cv2.imread(c1)/255.0
    compB = cv2.imread(c2)/255.0

    bgTest =  cv2.imread(newBg)/255.0

    fg, alpha = findMatting(compA, compB, backA, backB)
    newBg = modifyBackgroundWithAlpha(bgTest, alpha)
    composedImg = newBg + fg

    orignal = np.hstack([backA, backB, compA, compB])
    modified = np.hstack([bgTest, fg, composedImg, np.zeros(compA.shape)])

    cv2.imshow('New Composite', np.vstack([orignal, modified]))



# Test 1: Sample1
work(
    '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample1/BackgroundImg01.jpg',
    '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample1/BackgroundImg02.jpg',
    '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample1/CompositeImg01.jpg',
    '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample1/CompositeImg02.jpg',
    '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample1/NewBackground01.jpg'
)

# Test 2: Sample2
# work(
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample2/BackgroungImg01.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample2/BackgroungImg02.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample2/CompositeImg01.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample2/CompositeImg02.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample2/NewBackground01.jpg'
# )

# Test 3: Sample3
# work(
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample3/BackgroundImg01.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample3/BackgroundImg02.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample3/CompositeImg01.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample3/CompositeImg02.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample3/NewBackground01.jpg'
# )

# Test 4: Sample4
# work(
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample4/BackgroundImg01.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample4/BackgroundImg02.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample4/CompositeImg01.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample4/CompositeImg02.jpg',
#     '/Users/Kopkap/Desktop/multimedia/Trigulation-project/Sample4/NewBackground01.jpg'
# )


cv2.waitKey(0)
cv2.destroyAllWindows()