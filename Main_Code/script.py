import json
import math
data = {}
counter = { 'Detection': 0, 'In': 0, 'Theta': 0, 'DelX': 0 } # Y axis (up/down), X axis (left/right)
direction = None
def send():
    b = Buffer(90)
    b.setData(json.dumps(counter).encode('utf-8'))
    node.io['out'].send(b)

def tracklet_removed(tracklet, coords2, roi2):
    coords1 = tracklet['coords']
    deltaX = coords2[0] - coords1[0]
    deltaY = coords2[1] - coords1[1]
    
    roi = tracklet['corner']
    x1 = roi.topLeft().x
    y1 = roi.topLeft().y
    x2 = roi.bottomRight().x
    y2 = roi.bottomRight().y
    
    x21 = roi2.topLeft().x
    y21 = roi2.topLeft().y
    x22 = roi2.bottomRight().x
    y22 = roi2.bottomRight().y
    
    deltaXbox = abs(x22 - x21) - abs(x2 - x1)
    deltaYbox = abs(y21 - y22) - abs(y1 - y2)
    
    if deltaX == 0:
        theta = math.pi
    else:
        theta = math.tan(deltaY/deltaX)

    if abs(deltaX) > abs(deltaY) and abs(deltaX) > THRESH_DIST_DELTA:
        counter["Detection"] += 1
        
        if 0 > deltaX and x21 < 0.1 and y22 > 0.8: 
            direction = "In" 
            counter[direction] += 1
        else:
            direction = "Unknown"
        node.warn(f"{x21, y22}")
        counter["DelX"] = deltaXbox
        counter["Theta"] = theta
        
        send()
        print('Car detected')
        node.warn(f"Vehicle drove {direction}")
        # node.warn("DeltaX: " + str(abs(deltaX)))
        '''
    elif abs(deltaY) > abs(deltaX) and abs(deltaY) > THRESH_DIST_DELTA:
        direction = "In" if 0 > deltaY else "down"
        counter[direction] += 1
        send()
        node.warn(f"Vehicle moved {direction}")
        #node.warn("DeltaY: " + str(abs(deltaY)))
        '''
    #else: node.warn("Invalid movement")

def get_centroid(roi):
    x1 = roi.topLeft().x
    y1 = roi.topLeft().y
    x2 = roi.bottomRight().x
    y2 = roi.bottomRight().y
    return ((x2-x1)/2+x1, (y2-y1)/2+y1)

# Send dictionary initially (all counters 0)
send()
testd = 0
while True:
    tracklets = node.io['tracklets'].get()
    testd += 1
    if testd%10 == 0:
        print('spript.py 10')
    for t in tracklets.tracklets:
        # If new tracklet, save its centroid
        print('T2')
        if t.status == Tracklet.TrackingStatus.NEW:
            data[str(t.id)] = {} # Reset
            data[str(t.id)]['coords'] = get_centroid(t.roi)
            data[str(t.id)]['corner'] = t.roi
        elif t.status == Tracklet.TrackingStatus.TRACKED:
            data[str(t.id)]['lostCnt'] = 0
        elif t.status == Tracklet.TrackingStatus.LOST:
            data[str(t.id)]['lostCnt'] += 1
            # If tracklet has been "LOST" for more than 10 frames, remove it
            if 10 < data[str(t.id)]['lostCnt'] and "lost" not in data[str(t.id)]:
                #node.warn(f"Tracklet {t.id} lost: {data[str(t.id)]['lostCnt']}")
                tracklet_removed(data[str(t.id)], get_centroid(t.roi), t.roi)
                data[str(t.id)]["lost"] = True
        elif (t.status == Tracklet.TrackingStatus.REMOVED) and "lost" not in data[str(t.id)]:
            tracklet_removed(data[str(t.id)], get_centroid(t.roi), t.roi)
            #node.warn(f"Tracklet {t.id} removed")