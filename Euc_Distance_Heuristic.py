import numpy as np
import glob
import os
import shutil


# Global Var
num_iframes = 130
feats_dir = 'D:\\Projects\\research-ms-loss\\resource\\datasets\\labeled_i-frames\\features\\'
features = open(feats_dir + "iframe_feats.txt", 'r').readlines()  # Loading Data
labels = open(feats_dir + "iframe_labels.txt", 'r').readlines()
features = [[float(l.rstrip()) for l in f.split(' ')] for f in features]  # Preprocessing Data
labels = [int(l[0]) for l in labels]
#### Goal: Heuristic clustering by Euclidean distance between 130 of R^512 image features
thresh = 1.01
sets_dict = {}  # bin for the clusters


def find_single_target(idx):
    for k in sets_dict.keys():
        if idx in sets_dict[k]:
            return k

def main():
    for i in range(len(features)):
        sets_dict.setdefault(i, [i])

    # Looking at all feature data loaded,
    for ki in sets_dict.keys():
        min_dist, min_idx = np.Inf, -1
        #### find a feature that is closest to the ki_th feature
        for kj in sets_dict.keys():
            if ki == kj:
                continue

            dist = np.linalg.norm(np.array(features[ki]) - np.array(features[kj]))
            if min_dist > dist:
                min_dist, min_idx = dist, kj
        ####
        #### if the closest feature is lesser than the threshold AND the reverse is not placed in dictionary already,
        if min_dist < thresh and ki not in sets_dict[min_idx]:
            #### and if
            if not sets_dict[ki] and not sets_dict[min_idx]:
                k1, k2 = find_single_target(ki), find_single_target(min_idx)
                if k1 != k2:
                    sets_dict[k1].extend(sets_dict[k2])
                    sets_dict[k2] = []
            elif not sets_dict[ki]:
                k1 = find_single_target(ki)
                sets_dict[k1].extend(sets_dict[min_idx])
                sets_dict[min_idx] = []
            elif not sets_dict[min_idx]:
                k2 = find_single_target(min_idx)
                sets_dict[k2].extend(sets_dict[ki])
                sets_dict[ki] = []
            else:
                sets_dict[ki].extend(sets_dict[min_idx])
                sets_dict[min_idx] = []

    print('Threshold:', thresh)
    print('Clusters:', sets_dict)
    print('Number of clusters:', len([cl for cl in sets_dict.values() if len(cl) > 0]))
    print('Cluster with more than one element:', [key for key in sets_dict.keys() if len(sets_dict[key]) > 1])
    print('Number of features in all clusters:', sum([len(v) for v in sets_dict.values()]))


    in_dirs = glob.glob('./labeled_i-frames/**/*.jpg', recursive=True)
    out_dir = './clustered_images/'
    try:
        shutil.rmtree(out_dir) if os.path.exists(out_dir) else None
    except:
        pass
    try:
        os.mkdir(out_dir)
    except:
        pass
    for key in sets_dict.keys():
        dir = out_dir + str(key) + '/'
        os.mkdir(dir) if not os.path.exists(dir) and len(sets_dict[key]) > 0 else None
        for val in sets_dict[key]:
            shutil.copyfile(in_dirs[val], dir+os.path.basename(in_dirs[val]))


if __name__ == '__main__':
    main()