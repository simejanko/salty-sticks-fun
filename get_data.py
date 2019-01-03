import cv2
import numpy as np

A3_WIDTH = 29.7
A3_HEIGHT = 42.0


def img_show(img, w=1280, h=720):
    """ Utility for plotting image. """
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', w, h)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segment_a3(img, threshold=60, margin=15, debug=False):
    """ Segments out the paper and determines px->cm conversion ratio."""
    # threshold and find external contours
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        img_show(img_thresh)

    # get the largest contour
    contour = max(contours, key=cv2.contourArea)

    # approximate contour to get a rectangle
    arc_len = cv2.arcLength(contour, True)
    contour_approx = cv2.approxPolyDP(contour, 0.1 * arc_len, True)
    assert contour_approx.shape == (4, 1, 2), "Didn't find a rectangle."

    if debug:
        img_copy = img.copy()
        cv2.drawContours(img_copy, [contour_approx], -1, (0, 255, 0), 5)
        img_show(img_copy)

    # determine the conversion ratio
    contour_approx = contour_approx.reshape(4, 2)
    segments = [contour_approx[i] - contour_approx[i - 1] for i in range(contour_approx.shape[0])]
    segment_lengths = np.sort(np.linalg.norm(segments, axis=1))
    a3_px_width = sum(segment_lengths[:2]) / 2
    a3_px_height = sum(segment_lengths[2:]) / 2
    conversion_ratio = (A3_WIDTH / a3_px_width + A3_HEIGHT / a3_px_height) / 2

    # segment out the paper
    c_x = np.sort(contour_approx[:, 0])
    c_y = np.sort(contour_approx[:, 1])
    img_cropped = img[c_y[1]+margin:c_y[2]-margin, c_x[1]+margin:c_x[2]-margin]
    return img_cropped, conversion_ratio


def get_a3_lengths(img_file, threshold=60, close_size=10, open_size=10, size_threshold=1000, margin=15, debug=False):
    """
    Computes lenghts of objects on a A3 paper.
    :param threshold: grayscale boundry.
    :param close_size: size for morphological close.
    :param open_size: size for morphological open.
    :param size_threshold: area filter threshold for contours.
    :param margin: margin for segmenting A3 paper
    :return: Lengths (cm) of objects on a A3 paper
    """
    img_full = cv2.imread(img_file)
    img, conversion_ratio = segment_a3(img_full, threshold=threshold, margin=margin, debug=debug)

    # threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)

    if debug:
        img_show(img_thresh)

    # morphological close followed by an open
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))

    if debug:
        img_show(img_thresh)

    # find countours and filter out the small ones
    _, contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= size_threshold]

    if debug:
        img_copy = img.copy()
        cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 5)
        rects = [np.int0(cv2.boxPoints(cv2.minAreaRect(c))) for c in contours]
        cv2.drawContours(img_copy, rects, -1, (0, 0, 255), 5)
        img_show(img_copy)

    # fit rotated rectangles to contours and get their length
    lengths = [max(cv2.minAreaRect(c)[1]) * conversion_ratio for c in contours]

    return lengths


if __name__ == '__main__':
    lengths = get_a3_lengths('data/no_drop.jpg', debug=True)
    print("Number of no drop sticks: {} \nLengths: {}".format(len(lengths), lengths))
    np.savetxt('data/no_drop', lengths, '%.2f')
    print()

    lengths = get_a3_lengths('data/drop_v2.jpg', debug=True, threshold=65)
    print("Number of drop sticks: {} \nLengths: {}".format(len(lengths), lengths))
    np.savetxt('data/drop', lengths, '%.2f')
    print()


