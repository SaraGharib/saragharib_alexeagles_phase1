import cv2 as cv
import numpy as np

# function to calculate the number of gear teeth by analyzing contours
def count_teeth(contours, min_area, max_area):
    count = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if min_area <= area <= max_area:
            count += 1
    return count

# Load and process the ideal gear image
ideal = cv.imread('samples/ideal.jpg')
gray_ideal = cv.cvtColor(ideal, cv.COLOR_BGR2GRAY)
ret, th1 = cv.threshold(gray_ideal, 90, 255, cv.THRESH_BINARY)
canny1 = cv.Canny(th1, 125, 175)
blur1 = cv.GaussianBlur(canny1, (1, 1), cv.BORDER_DEFAULT)

# Iterate over each sample gear image
for n in range(2, 7):
    sample = cv.imread(f'samples/sample{n}.jpg')
    gray_sample = cv.cvtColor(sample, cv.COLOR_BGR2GRAY)
    ret, th2 = cv.threshold(gray_sample, 90, 255, cv.THRESH_BINARY)
    canny2 = cv.Canny(th2, 125, 175)
    blur2 = cv.GaussianBlur(canny2, (1, 1), cv.BORDER_DEFAULT)

    # Compute the difference between the ideal gear and the sample gear
    diff = cv.bitwise_xor(blur1, blur2)
    contours, hierarchies = cv.findContours(diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Initialize counters and status flags
    worn = []
    broken = []
    missing_teeth = False
    missing_inner_diameter = False
    inner_diameter = None
    d = 'identical inner diameters'

    for contour in contours:
        area = cv.contourArea(contour)
        (x, y), radius = cv.minEnclosingCircle(contour)
        
        if area >= 5:  # Filter out small noise
            if area <= 500:
                worn.append(contour)
            elif 500 < area <= 2000:
                broken.append(contour)
            elif area > 2000:
                if radius < 10:  # If the radius is small, it's likely a missing inner diameter
                    missing_inner_diameter = True
                    d = 'missing inner diameter'
                else:
                    d = 'larger inner diameter than ideal sample'

            if inner_diameter is None or radius < inner_diameter:
                inner_diameter = radius

    # Count teeth to detect missing ones
    ideal_teeth_count = count_teeth(cv.findContours(blur1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0], 500, 2000)
    sample_teeth_count = count_teeth(cv.findContours(blur2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0], 500, 2000)

    if sample_teeth_count < ideal_teeth_count:
        missing_teeth = True

    # Display the analysis results for each sample
    print(f'FOR SAMPLE NO. {n}:')
    if missing_teeth:
        print(f"Missing teeth detected (Ideal: {ideal_teeth_count}, Sample: {sample_teeth_count})")
    else:
        print(f'{len(worn)} worn teeth')
        print(f'{len(broken)} broken teeth')
    print(d)
    if inner_diameter is not None and not missing_inner_diameter:
        print(f'Inner diameter: {inner_diameter * 2:.2f} pixels')
    else:
        print('Missing or incorrect inner diameter')

    # Optionally visualize the results by drawing contours
    contour_img = sample.copy()
    cv.drawContours(contour_img, worn, -1, (0, 255, 0), 2)  # Green for worn teeth
    cv.drawContours(contour_img, broken, -1, (0, 0, 255), 2)  # Red for broken teeth
    cv.imshow(f'Analysis Sample {n}', contour_img)

cv.waitKey(0)
cv.destroyAllWindows()
