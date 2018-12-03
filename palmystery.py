import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import sqrt

image = 'palm3.jpg'

# Read in file and convert to maximized B&W version
raw = cv2.imread(image)
ref = cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
ref = np.float32(ref)
for pixel in ref:
    if pixel[0] >= 210 and pixel[1] >= 210 and pixel[2] >= 210:
        pixel = [255, 255, 255]
    else:
        pixel = [0, 0, 0]

# Run corner detection
corners = cv2.goodFeaturesToTrack(ref, 100, 0.01, 10)
corners = np.int0(corners)

# Parse corners
parsed = []
for corner in corners:
    parsed.append(corner[0])
parsed = np.asarray(parsed)

# Run K-means
kmeans = KMeans(n_clusters=3).fit(parsed)

# Place points into their respective clusters (lines)
line_0 = []
line_1 = []
line_2 = []
predicted = kmeans.predict(parsed)
for i in range(len(predicted)):
    line = predicted[i]
    if line == 0:
        line_0.append(parsed[i])
    if line == 1:
        line_1.append(parsed[i])
    if line == 2:
        line_2.append(parsed[i])

# Select 'start' and 'end' points for each line
line_0_start = [min(line_0, key=lambda point:point[0])[0], min(line_0, key=lambda point:point[1])[1]]
line_0_end = [max(line_0, key=lambda point:point[0])[0], max(line_0, key=lambda point:point[1])[1]]
line_1_start = [min(line_1, key=lambda point:point[0])[0], min(line_1, key=lambda point:point[1])[1]]
line_1_end = [max(line_1, key=lambda point:point[0])[0], max(line_1, key=lambda point:point[1])[1]]
line_2_start = [min(line_2, key=lambda point:point[0])[0], min(line_2, key=lambda point:point[1])[1]]
line_2_end = [max(line_2, key=lambda point:point[0])[0], max(line_2, key=lambda point:point[1])[1]]

# Calculate attributes for each line
def distance(start, end):
    x1, y1 = start[0], start[1]
    x2, y2 = end[0], end[1]
    dist = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist
line_0_height = abs(line_0_end[1] - line_0_start[1])
line_1_height = abs(line_1_end[1] - line_1_start[1])
line_2_height = abs(line_2_end[1] - line_2_start[1])
line_0_distance = distance(line_0_start, line_0_end)
line_1_distance = distance(line_1_start, line_1_end)
line_2_distance = distance(line_2_start, line_2_end)

# ~Palmistry~
# Life Line (line_0)
if line_0_height > 350:
    height_str_0 = 'You have a tall, curving life line indicating strength, enthusiasm, vitality (#RahulA #ifyouknowyouknow)'
elif line_0_height < 300:
    height_str_0 = 'You have a thin, flat life line indicating that you are easily manipulated by others (they said CS61A would be easy)'
else:
    height_str_0 = 'You have a moderate life line; be cautious when it comes to relationships (beware eecs bois and abgs)'
if line_0_distance > 400:
    distance_str_0 = 'You have a long, swirling life line foretelling a sudden change in lifestyle (hope you like being 1 sd below the mean)'
elif line_0_distance < 350:
    distance_str_0 = 'You have a short, straight life line indicating that you are very naive and gullible (@everyone on WWPD)'
else:
    distance_str_0 = 'You have a average life line meaning that you are often tired (welcome to CS61A)'
# Head Line (line 1)
if line_1_height > 325:
    height_str_1 = 'You have a towering, arching head line indicating creativity (did you enter in the Scheme Art Contest?)'
elif line_1_height < 275:
    height_str_1 = 'You have a narrow, stubby head line indicating that you think realistically (just take the L)'
else:
    height_str_1 = 'You have a standard head line meaning you have a short attention span (watch lectures at minimum 2.5x speed)'
if line_1_distance > 375:
    distance_str_1 = 'You have a tall head line indicating inconsistencies in thought (environment diagrams are a struggle for you)'
elif line_1_distance < 325:
    distance_str_1 = 'You have a tiny, wavering head line indicating that you prefer physical achievements over mental ones (would take pride in writing 10000 lines of code, even if they all error)'
else:
    distance_str_1 = 'You have a medium length head line which indicates that you are coming out of an emotional crisis (the semester is almost over)'
# Heart Line
if line_2_height > 425:
    height_str_2 = 'You have a long heart line indicating that you are content with love life (you are that guy/girl on subtle asian dating)'
elif line_2_height < 375:
    height_str_2 = 'You have a faint heart line indicating that you fall in love easily (you liked Scheme)'
else:
    height_str_2 = 'You have a medium thickness heart line indicating that you are less interested in romance (only code, no bae)'
if line_2_distance > 450:
    distance_str_2 = 'You have a large heart line meaning you freely express emotions and feelings (you cried when you saw MT1...and MT2)'
elif line_2_distance < 400:
    distance_str_2 = 'You have a small heart line indicating many relationships and lovers, but an absence of serious ones (the effect of MT2)'
else:
    distance_str_2 = 'You have a typically sized heart line meaning that you are a very happy person (congrats on finishing CS61A)'
def output(h0, d0, h1, d1, h2, d2):
    print(".........................................................................................................................................................")
    print("The Life Line reflects physical health, general well being, and major life changes (for example, cataclysmic events, physical injuries, and relocation).")
    print("Your reading is:")
    print(h0)
    print(d0)
    print(".........................................................................................................................................................")
    print("The Head Line represents learning style, communication style, intellectualism, and thirst for knowledge. It is traditionally a very significant line.")
    print("Your reading is:")
    print(h1)
    print(d1)
    print(".........................................................................................................................................................")
    print("The Heart Line is believed to indicate emotional stability, romantic perspectives, depression, and cardiac health.")
    print("Your reading is:")
    print(h2)
    print(d2)
    print(".........................................................................................................................................................")
    print("Special Thanks to Wiki How for These Descriptions!")
output(height_str_0, distance_str_0, height_str_1, distance_str_1, height_str_2, distance_str_2)

# Plot corners
for corner in corners:
    x, y = corner.ravel() # Extract X and Y
    cv2.circle(raw, (x, y), 3, 255, -1) # Plot each corner
cv2.imshow('Corner', raw)

# Plot K-means centroids
ct = plt.imread(image)
implot = plt.imshow(ct)
plt.plot([kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[2][0]], [kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[1][1], kmeans.cluster_centers_[2][1]],'o')
plt.show()

# Window management
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()