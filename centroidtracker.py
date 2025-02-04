from collections import OrderedDict
import time
import numpy as np
from scipy.spatial import distance

class CentroidTracker:
    def __init__(self, max_disappeared=50, expiration_time=1.0):
        # Store the next available object ID
        self.next_object_id = 0
        # Dictionary to store object ID and its centroid
        self.objects = OrderedDict()
        # Dictionary to store the last timestamp an object was seen
        self.last_seen = OrderedDict()
        # Store the number of frames an object has been marked as disappeared
        self.disappeared = OrderedDict()
        # Expiration time in seconds
        self.expiration_time = expiration_time

    def register(self, centroid):
        """Register a new object with a centroid and timestamp."""
        self.objects[self.next_object_id] = centroid
        self.last_seen[self.next_object_id] = time.time()  # Store the current timestamp
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Deregister an object by removing it from tracking."""
        del self.objects[object_id]
        del self.last_seen[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        Update centroids based on the new bounding boxes.
        rects: List of bounding box tuples [(startX, startY, endX, endY), ...]
        """
        current_time = time.time()

        # If no objects are detected, increment disappearance count
        if len(rects) == 0:
            for object_id in list(self.last_seen.keys()):
                if (current_time - self.last_seen[object_id]) > self.expiration_time:
                    self.deregister(object_id)
            return self.objects

        # Convert bounding boxes to centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        # If no objects are currently tracked, register all input centroids
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Get existing object IDs and their centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance between existing centroids and new centroids
            D = distance.cdist(np.array(object_centroids), input_centroids)

            # Find the smallest distance between new and existing objects
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)

            # Track assignments
            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.last_seen[object_id] = current_time  # Update last seen timestamp
                self.disappeared[object_id] = 0  # Reset disappeared count

                used_rows.add(row)
                used_cols.add(col)

            # Deregister objects not updated within expiration time
            for object_id in list(self.objects.keys()):
                if object_id not in [object_ids[row] for row in used_rows]:
                    if (current_time - self.last_seen[object_id]) > self.expiration_time:
                        self.deregister(object_id)

            # Register new objects
            for col in set(range(D.shape[1])) - used_cols:
                self.register(input_centroids[col])

        return self.objects
