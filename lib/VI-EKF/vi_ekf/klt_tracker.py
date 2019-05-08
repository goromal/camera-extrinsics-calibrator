import cv2
import numpy as np

class KLT_tracker:
    def __init__(self, num_features=25, show_image=False, radius=50):

        self.prev_image = []
        self.initialized = False

        self.plot_matches = show_image

        self.num_features = num_features
        self.feature_nearby_radius = radius

        self.features = np.zeros((self.num_features, 2, 1))
        self.ids = np.zeros(self.num_features)

        self.feature_params = dict(qualityLevel=0.3,
                                   minDistance=self.feature_nearby_radius,
                                   blockSize=7)

        self.lk_params = dict(winSize  = (15,15),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.color = np.random.randint(0, 255, (self.num_features, 3))
        self.next_feature_id = 0
        self.video = cv2.VideoWriter('plots/tracker_output.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 30, (640, 480))

    def load_image(self, img, t=None):
        # Load Image
        # If first time initialize the feature
        if not self.initialized:
            # Capture a bunch of new features
            self.features = cv2.goodFeaturesToTrack(img, mask=None, maxCorners=self.num_features, **self.feature_params)

            # Set the first set of indexes
            self.ids = np.array([i+1 for i in range(len(self.features))])
            self.next_feature_id = self.num_features+1

            # TODO: handle the case that we don't get enough features

            # Save off the image
            self.prev_image = img

            # Tell the tracker we have initialized
            self.initialized = True

        # Otherwise, we've already initialized
        else:
            # calculate optical flow
            new_features, st, err = cv2.calcOpticalFlowPyrLK(self.prev_image, img, self.features, None, **self.lk_params)

            st = st[:, 0]

            # Select good points
            matched_features = new_features[st == 1]

            # good_old = [self.features[st == 1], self.ids[st==1]]

            # Now update the previous frame and previous points
            self.prev_image = img.copy()
            self.features = matched_features.reshape(-1, 1, 2)


            # Drop lost features from the tracker
            self.ids = self.ids[st == 1]


            # If we are missing points, collect new ones
            if len(matched_features) < self.num_features:
                # First, create a mask around the current points
                current_point_mask = np.ones_like(img)
                for point in matched_features:
                    a, b = point.ravel()
                    cv2.circle(current_point_mask, (a, b), self.feature_nearby_radius, 0, thickness=-1, lineType=0)

                # Now find a bunch of points, not in the mask
                num_new_features = self.num_features - len(matched_features)
                new_features = cv2.goodFeaturesToTrack(img, mask=current_point_mask, maxCorners=num_new_features, **self.feature_params)

                # if we found new features to track
                if new_features is not None:
                    # Append the new features to the tracker feature list
                    self.features = np.concatenate((self.features, new_features), axis=0)
                    self.ids = np.concatenate((self.ids, np.array([i + self.next_feature_id for i in range(len(new_features))])), axis=0)

                    # Increment the index of what the next feature it is
                    self.next_feature_id = num_new_features + self.next_feature_id

        # If we are debugging, plot the points
        if self.plot_matches:
            # Convert the image to color
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # draw the features and ids
            for id, point in zip(self.ids, self.features):
                x, y = point.ravel()
                cv2.circle(img, (x, y), 5, self.color[id % self.num_features].tolist(), -1)
                cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

            if t is not None:
                cv2.putText(img, "t={:.2f}".format(t), (560, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 100))

            cv2.imshow("Image window", img)
            self.video.write(img)
            cv2.waitKey(1)

        return self.features, self.ids

    def close(self):
        self.video.release()

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    tracker = KLT_tracker(25, True)

    while True:
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tracker.load_image(gray)
