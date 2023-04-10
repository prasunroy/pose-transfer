import numpy as np
import torch


class SSDScore(object):
    
    def __init__(self, precision='fp32', use_gpu=True):
        self.precision = precision
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision, trust_repo=True, verbose=False)
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils', trust_repo=True, verbose=False)
        if use_gpu and torch.cuda.is_available():
            self.model.to('cuda')
        self.model.eval()
    
    def eval(self, images, batch_size=32, class_label=1, verbose=False):
        scores = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_inputs = [self.utils.prepare_input(image) for image in batch_images]
            batch_tensor = self.utils.prepare_tensor(batch_inputs, self.precision == 'fp16')
            with torch.no_grad():
                batch_detections = self.model(batch_tensor)
            results_per_batch_input = self.utils.decode_results(batch_detections)
            for result in results_per_batch_input:
                class_scores = result[2][result[1] == class_label]
                if len(class_scores) == 0:
                    scores.append(0)
                else:
                    scores.append(max(class_scores))
                if verbose:
                    print(f'\r[SSD SCORE] Progress: {len(scores)*100.0/len(images):3.0f}% |',
                          f'mean: {np.mean(scores):.3f} | std: {np.std(scores):.3f} |', end='')
        if verbose:
            print('')
        return (np.mean(scores), np.std(scores))
