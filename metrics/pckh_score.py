import numpy as np


class PCKhScore(object):
    
    def __init__(self, head_keypoint_ids=[0, 1, 14, 15, 16, 17]):
        self.head_keypoint_ids = head_keypoint_ids
    
    def _estimate_head_size(self, keypoints):
        head_w, head_h = -1, -1
        keypoints = keypoints.reshape(-1, 2)
        selected_keypoints = []
        for keypoint_id in self.head_keypoint_ids:
            keypoint = keypoints[keypoint_id]
            if keypoint[0] != -1 and keypoint[1] != -1:
                selected_keypoints.append(keypoint)
        if len(selected_keypoints) >= 2:
            selected_keypoints = np.asarray(selected_keypoints)
            xmin = selected_keypoints[:, 0].min()
            xmax = selected_keypoints[:, 0].max()
            ymin = selected_keypoints[:, 1].min()
            ymax = selected_keypoints[:, 1].max()
            head_w = xmax - xmin
            head_h = ymax - ymin
        return head_w, head_h
    
    def _count_valid(self, keypoints):
        num_valid = 0
        keypoints = keypoints.reshape(-1, 2)
        for x, y in keypoints:
            if x != -1 and y != -1:
                num_valid += 1
        return num_valid
    
    def _iscorrect(self, x, y, pred_x, pred_y, head_size, alpha=0.5):
        if x == -1 or y == -1 or pred_x == -1 or pred_y == -1:
            return 0
        if abs(x - pred_x) < head_size[0] * alpha and abs(y - pred_y) < head_size[1] * alpha:
            return 1
        else:
            return 0
    
    def _count_correct(self, actual_keypoints, predicted_keypoints, head_size, alpha=0.5):
        num_correct = 0
        actual_keypoints = actual_keypoints.reshape(-1, 2)
        predicted_keypoints = predicted_keypoints.reshape(-1, 2)
        for (x, y), (pred_x, pred_y) in zip(actual_keypoints, predicted_keypoints):
            num_correct += self._iscorrect(x, y, pred_x, pred_y, head_size, alpha)
        return num_correct
    
    def eval(self, all_actual_keypoints, all_predicted_keypoints, alpha=0.5, verbose=False):
        num_valid = 0
        num_correct = 0
        for i, (actual_keypoints, predicted_keypoints) in enumerate(zip(all_actual_keypoints, all_predicted_keypoints)):
            head_size = self._estimate_head_size(actual_keypoints)
            if head_size[0] == -1 or head_size[1] == -1:
                continue
            num_valid += self._count_valid(actual_keypoints)
            num_correct += self._count_correct(actual_keypoints, predicted_keypoints, head_size, alpha)
            if verbose:
                print(f'\r[PCKh SCORE] Progress: {(i+1)*100.0/len(all_predicted_keypoints):3.0f}% |',
                      f'PCKh: {num_correct}/{num_valid} = {num_correct * 1.0 / num_valid:.2f} |', end='')
        if verbose:
            print('')
        
        pckh = num_correct * 1.0 / num_valid
        return (pckh, num_correct, num_valid)
