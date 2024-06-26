'''
Description: Cell-tissue Graph Pathological Analysis Model (CTG-PAM)
'''
import os
import numpy as np
from matplotlib import image as mi
from utils import *
from model import LightGCN, CellMLP
from torch.utils.data import Dataset
import torch
import detector
from skimage import measure as skm
import math
import networkx as nx
PI = math.pi
torch.manual_seed(42)

def calculate_confusion_matrix(predictions, labels):
    valid_indices = labels != -1
    filtered_predictions = predictions[valid_indices]
    filtered_labels = labels[valid_indices]
    TP = np.sum((filtered_predictions == 1) & (filtered_labels == 1))
    TN = np.sum((filtered_predictions == 0) & (filtered_labels == 0))
    FP = np.sum((filtered_predictions == 1) & (filtered_labels == 0))
    FN = np.sum((filtered_predictions == 0) & (filtered_labels == 1))
    return TP, TN, FP, FN

def generate_red_colors(n):
    colors = []
    for i in range(n):
        r = 1.0  
        g = np.random.uniform(0, 0.3)  
        b = np.random.uniform(0, 0.3)  
        colors.append((r, g, b))
    return colors


class cells_dataset():
    def __init__(self, path, cellgraph_n=20, tissuegraph_n=2, cth=50, tth=5, ifdetector=True):
        self.dataset_path = path
        self.slice_list = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
        self.cell_n = cellgraph_n
        self.tissue_n = tissuegraph_n
        self.cell_th = cth
        self.tissue_th = tth
        self.cell_detector = None
        self.tissue_detector = None
        if ifdetector:
            from stardist.models import StarDist2D
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            model = StarDist2D.from_pretrained('2D_versatile_he')
            sam = sam_model_registry["vit_h"](checkpoint="/mnt/data/ctg_dataset/sam_vit_h_4b8939.pth")
            device = "cuda:3"
            sam = sam.to(device)
            mask_generator = SamAutomaticMaskGenerator(sam, 
            points_per_side=32, points_per_batch=16)
            self.cell_detector = model
            self.tissue_detector = mask_generator
    def cell_generate(self):
        for name in self.slice_list:
            img = mi.imread(os.path.join(self.dataset_path, name, "img.png"))
            cell_mask = detector.cell_detector_from_model(img, self.cell_detector)
            np.save(os.path.join(self.dataset_path, name, "cell_map.npy"), cell_mask)
    def tissue_generate(self):
        for i, name in enumerate(self.slice_list):
            if "txt" in name:
                continue
            print("%d/%d"%(i, len(self.slice_list)))
            img = mi.imread(os.path.join(self.dataset_path, name, "img.png"))
            tissue_mask = detector.tissue_detector_from_model(img, self.tissue_detector)
            np.save(os.path.join(self.dataset_path, name, "tissue_map.npy"), tissue_mask)
    def cell_feature_init(self):
        for i, slice_name in enumerate(self.slice_list):
            cell_feature = self.node_feature_slice(slice_name)
            np.save(os.path.join(self.dataset_path, slice_name, "cell_feature.npy"), cell_feature)
    def tissue_feature_init(self):
        for i, slice_name in enumerate(self.slice_list):
            cell_feature = self.node_feature_slice(slice_name, False)
            np.save(os.path.join(self.dataset_path, slice_name, "tissue_feature.npy"), cell_feature)
    def distmatrix_init(self):
        for i, slice_name in enumerate(self.slice_list):
            cc_dist, ct_dist = self.dist_slice(slice_name)
            np.save(os.path.join(self.dataset_path, slice_name, "cc_dist.npy"), cc_dist)
            np.save(os.path.join(self.dataset_path, slice_name, "ct_dist.npy"), ct_dist)      
    def dist_slice(self, slice_name):
        if slice_name not in self.slice_list:
            raise ValueError
        folder = os.path.join(self.dataset_path, slice_name)
        cell_map_npy = "cell_map.npy"
        tissue_map_npy = "tissue_map.npy"
        cell_map = np.load(os.path.join(folder, cell_map_npy)).astype("int32")
        tissue_map = np.load(os.path.join(folder, tissue_map_npy)).astype("int32")
        cell_num = int(np.max(cell_map))
        tissue_num = int(np.max(tissue_map))
        cell_location = np.zeros((cell_num, 2))
        tissue_location = np.zeros((tissue_num, 2))
        cell_area = np.zeros((cell_num, 1))
        tissue_area = np.zeros((tissue_num, 1))
        cell_features = skm.regionprops(cell_map)
        tissue_features = skm.regionprops(tissue_map)
        for i in range(int(cell_num)):
            cell_location[i, 0] = cell_features[i]["centroid"][0]
            cell_location[i, 1] = cell_features[i]["centroid"][1]
            cell_area[i] = cell_features[i]["area"]
        for i in range(int(tissue_num)):
            tissue_location[i, 0] = tissue_features[i]["centroid"][0]
            tissue_location[i, 1] = tissue_features[i]["centroid"][1]
            tissue_area[i] = tissue_features[i]["area"]
        cc_dist = location_to_dist(cell_location, cell_location)
        ct_dist = location_to_dist(cell_location, tissue_location)
        cell_area_repeat = cell_area.repeat(cell_num, 1)
        cell_r = np.sqrt(cell_area_repeat) / PI
        cc_dist_new = cc_dist - cell_r - cell_r.T
        cc_dist_new[cc_dist_new<0] = 0
        tissue_area = tissue_area.repeat(cell_num, 1)
        cell_area_repeat = cell_area.repeat(tissue_num, 1)
        cell_r = np.sqrt(cell_area_repeat) / PI
        tissue_r = np.sqrt(tissue_area) / PI
        ct_dist_new = ct_dist - cell_r - tissue_r.T 
        return cc_dist_new, ct_dist_new
    def node_feature_slice(self, slice_name, is_cell=True):
        if slice_name not in self.slice_list:
            raise ValueError
        folder = os.path.join(self.dataset_path, slice_name)
        if is_cell:
            map_npy = "cell_map.npy"
        else:
            map_npy = "tissue_map.npy"
        cell_map = np.load(os.path.join(folder, map_npy)).astype("int32")
        img = mi.imread(os.path.join(folder, "img.png"))[:, :, :3]
        img = np.mean(img, axis=2)
        cell_num = np.max(cell_map)
        cell_feature_slice = None
        feature = skm.regionprops(cell_map, img)
        feature_dim = 8
        cell_num = len(feature)
        cell_feature_slice = np.zeros((cell_num, feature_dim))
        for i in range(cell_num):
            cell_feature = feature[i]
            cell_feature_slice[i, 0] = cell_feature["intensity_mean"]
            cell_feature_slice[i, 1] = cell_feature["intensity_std"] ** 2
            area = cell_feature["area"]
            perimeter = cell_feature["perimeter"]
            cell_feature_slice[i, 2] = area
            cell_feature_slice[i, 3] = perimeter
            cell_feature_slice[i, 4] = perimeter ** 2 / area
            cell_feature_slice[i, 5] = cell_feature["major_axis_length"]
            cell_feature_slice[i, 6] = cell_feature["minor_axis_length"]
            cell_feature_slice[i, 7] = cell_feature["eccentricity"] 
        return cell_feature_slice
    def cell_adjust_feature_slice(self, slice_name):
        lightgcn_model = LightGCN(K=3)
        if slice_name not in self.slice_list:
            raise ValueError
        folder = os.path.join(self.dataset_path, slice_name)
        ct_dist = np.load(os.path.join(folder, "ct_dist.npy"))
        cc_dist = np.load(os.path.join(folder, "cc_dist.npy"))
        c_feature = np.load(os.path.join(folder, "cell_feature.npy"))
        t_feature = np.load(os.path.join(folder, "tissue_feature.npy"))
        cc_adjust_feature = np.zeros(c_feature.shape)
        ct_adjust_feature = np.zeros(c_feature.shape)
        cc_sort = np.argsort(cc_dist)
        cc_sort = cc_sort[:, :self.cell_n]
        ct_sort = np.argsort(ct_dist)
        ct_sort = ct_sort[:, :self.tissue_n]
        m = cc_sort.shape[0]
        for i in range(m):
            cell_sort = cc_sort[i, :]
            tissue_sort = ct_sort[i, :]
            X_cell = c_feature[cell_sort]
            sub_cc_dist = cc_dist[cell_sort].T[cell_sort].T
            A_cell = (sub_cc_dist<self.cell_th)*1 - np.eye(self.cell_n)
            D_cell = np.diag(A_cell.sum(axis=1))
            E_cell = lightgcn_model.compute(A_cell, D_cell, X_cell)
            cell_feature = E_cell[0, :]
            X_tissue = t_feature[tissue_sort]
            X_tissue = np.vstack((c_feature[i, :], X_tissue))
            sub_ct_dist = ct_dist[i, :][tissue_sort]
            sub_ct_dist = (sub_ct_dist<self.tissue_th)*1
            A_tissue = np.zeros((self.tissue_n+1, self.tissue_n+1))
            A_tissue[1:, 0] = sub_ct_dist
            A_tissue = A_tissue.T
            A_tissue[1:, 0] = sub_ct_dist
            D_tissue = np.diag(A_tissue.sum(axis=1))
            E_tissue = lightgcn_model.compute(A_tissue, D_tissue, X_tissue)
            tissue_feature = E_tissue[0, :]
            cc_adjust_feature[i, :] = cell_feature
            ct_adjust_feature[i, :] = tissue_feature
        os.makedirs(os.path.join(folder, "cc_adjust_feature"), exist_ok=True)     
        os.makedirs(os.path.join(folder, "ct_adjust_feature"), exist_ok=True)     
        np.save(os.path.join(folder, "cc_adjust_feature", "cc_adjust_feature_cth_%d_n_%d.npy"%(self.cell_th, self.cell_n)), cc_adjust_feature)
        np.save(os.path.join(folder, "ct_adjust_feature", "ct_adjust_feature_tth_%d_n_%d.npy"%(self.tissue_th, self.tissue_n)), ct_adjust_feature)
    def cell_adjust_feature_init(self):
        for i, slice_name in enumerate(self.slice_list):
            self.cell_adjust_feature_slice(slice_name)
    def save_png_map(self):
        for i, slice_name in enumerate(self.slice_list):
            print(i, len(self.slice_list))
            folder = os.path.join(self.dataset_path, slice_name)
            npynames = ["cell_map", "tissue_map"]
            for name in npynames:
                npy = np.load(os.path.join(folder, name+".npy"))
                mi.imsave(os.path.join(folder, name+".png"), norm(npy), cmap="cool")
    def label_init(self):
        for i, slice_name in enumerate(self.slice_list):
            print(i, len(self.slice_list))
            cell_label = self.cell_label_slice(slice_name)
            folder = os.path.join(self.dataset_path, slice_name)
            np.save(os.path.join(folder, "cell_label.npy"), cell_label)
    def plot_cell_mask_contours(self, slice_name, predict=False, predict_result=None):
        name = slice_name
        img = mi.imread(os.path.join(self.dataset_path, name, "img.png"))[:, :, :3]
        cell_mask = np.load(os.path.join(self.dataset_path, name, "cell_map.npy"))
        if predict:
            cell_label = predict_result
        else:
            cell_label = np.load(os.path.join(self.dataset_path, name, "cell_label.npy"))
            cell_label = cell_label[:, 0]
        label_map = np.zeros_like(cell_mask) - 1
        non_zero_indices = cell_mask > 0
        label_map[non_zero_indices] = cell_label[cell_mask[non_zero_indices] - 1]
        c = ["red", "blue", "green"]
        c[0] = (1, 0, 0)
        c[2] = (0, 1, 0)
        c[1] = (0, 0, 1)
        for i, color in enumerate(c):
            img[label_map==i] = color
        if predict:
            mi.imsave(os.path.join(self.dataset_path, name, "cell_predict.png"), img)
        else:
            mi.imsave(os.path.join(self.dataset_path, name, "cell_label.png"), img)
        plt.close()
    def plot_cell_mask_groups(self, slice_name, groups, cell_index):
        name = slice_name
        img = mi.imread(os.path.join(self.dataset_path, name, "img.png"))[:, :, :3]
        cell_mask = np.load(os.path.join(self.dataset_path, name, "cell_map.npy"))
        groups_n = len(groups)
        red_colors = generate_red_colors(groups_n)
        c = 0
        for group, color in zip(groups, red_colors):
            for cell in group:
                img[cell_mask==(cell_index[cell]+1)] = color
        mi.imsave(os.path.join(self.dataset_path, name, "cell_groups.png"), img)
        plt.close()
    def cell_label_slice(self, slice_name):
        if slice_name not in self.slice_list:
            raise ValueError
        folder = os.path.join(self.dataset_path, slice_name)
        cell_map = np.load(os.path.join(folder, "cell_map.npy"))
        if "gla_mask.png" in os.listdir(folder):
            gla_mask = (mi.imread(os.path.join(folder, "gla_mask.png"))[:,:,0]>0)*1
        else:
            gla_mask = np.zeros_like(cell_map)
        if "lym_mask.png" in os.listdir(folder):
            lym_mask = (mi.imread(os.path.join(folder, "lym_mask.png"))[:,:,0]>0)*1
        else:
            lym_mask = np.zeros_like(cell_map)
        cell_n = cell_map.max()
        cell_label = np.zeros((cell_n, 1))
        for i in range(cell_n):
            cell = (cell_map==(i+1))
            label = 0
            if lym_mask[cell].sum() > 0:
                label = 1
            elif gla_mask[cell].sum() > 0:
                label = 2
            cell_label[i] = label
        return cell_label
    def create_dataset(self):
        dataset = []
        for i, slice_name in enumerate(self.slice_list):
            folder = os.path.join(self.dataset_path, slice_name)
            files = ["cell_feature.npy", os.path.join("cc_adjust_feature", "cc_adjust_feature_cth_%d_n_%d.npy"%(self.cell_th, self.cell_n)), os.path.join("ct_adjust_feature", "ct_adjust_feature_tth_%d_n_%d.npy"%(self.tissue_th, self.tissue_n)), "cell_label.npy"]
            f = files[0]
            data = np.load(os.path.join(folder, f))
            for i in range(3):
                f = files[i+1]
                feature = np.load(os.path.join(folder, f))
                data = np.hstack((data, feature))
            dataset.append(data)
        return dataset
    def create_dataset_slice(self, slice_name):
        dataset = []
        folder = os.path.join(self.dataset_path, slice_name)
        files = ["cell_feature.npy", os.path.join("cc_adjust_feature", "cc_adjust_feature_cth_%d_n_%d.npy"%(self.cell_th, self.cell_n)), os.path.join("ct_adjust_feature", "ct_adjust_feature_tth_%d_n_%d.npy"%(self.tissue_th, self.tissue_n)), "cell_label.npy"]
        f = files[0]
        data = np.load(os.path.join(folder, f))
        for i in range(3):
            f = files[i+1]
            feature = np.load(os.path.join(folder, f))
            data = np.hstack((data, feature))
        dataset.append(data)
        return dataset    
    def cell_classification(self, model_path, visualization=False, mlp_n=50, feature_type="ctg", gpu="cuda:0"):
        if feature_type == "ctg":
            input_d = 24
        elif feature_type == "ccg":
            input_d = 16
        net = CellMLP(mlp_n, input_d)
        net.to(gpu)
        net.load_state_dict(torch.load(model_path, map_location=gpu), strict=True)
        net.eval()
        result_cm = {"tp":[], "tn":[], "fp":[], "fn":[], "slice_name":[]}
        tp_sum, tn_sum, fp_sum, fn_sum = 0, 0, 0, 0
        for i, slice_name in enumerate(self.slice_list):
            cell_dataset = np.vstack(self.create_dataset_slice(slice_name)).astype("float32")
            cell_data = cell_dataset[:, :-(25-input_d)]
            cell_label = cell_dataset[:, -1]
            cell_data = torch.tensor(cell_data).to(gpu)
            output = net(cell_data)
            predict = torch.max(output, 1)[1].cpu().data.numpy().squeeze() 
            stp, stn, sfp, sfn = calculate_confusion_matrix(predict, cell_label-1)
            result_cm["slice_name"].append(slice_name)
            result_cm["tp"].append(stp)
            result_cm["tn"].append(stn)
            result_cm["fp"].append(sfp)
            result_cm["fn"].append(sfn)
            tp_sum += stp
            tn_sum += stn
            fp_sum += sfp
            fn_sum += sfn
            cell_index = np.where(predict==0)
            cc_dist_mat = np.load(os.path.join(self.dataset_path, slice_name, "cc_dist.npy"))
            cc_dist_mat = cc_dist_mat[cell_index]
            cc_dist_mat = cc_dist_mat.T[cell_index]
            cc_dist_mat = cc_dist_mat.T
            cc_adjust = (cc_dist_mat < 30) * 1
            lym_graph = nx.from_numpy_array(cc_adjust)
            connected_subgraphs = list(nx.connected_components(lym_graph))
            groups_more_than_50 = []
            for i, subgraph in enumerate(connected_subgraphs):
                if len(list(subgraph)) > 50:
                    groups_more_than_50.append(list(subgraph))
            if len(groups_more_than_50) > 0:
                print("The diagnosis of %s is SS."%slice_name)
            else:
                print("The diagnosis of %s is nSS."%slice_name)

            if visualization:
                self.plot_cell_mask_contours(slice_name, True, predict)
                self.plot_cell_mask_groups(slice_name, groups_more_than_50, cell_index[0])        
        acc = (tp_sum+tn_sum)/(tp_sum+tn_sum+fn_sum+fp_sum)
        sen = (tp_sum)/(tp_sum+fn_sum)
        spc = (tn_sum)/(tn_sum+fp_sum)
        print("accuracy: ", acc)
        print("sensitivity for lymphocyte / specificity for other: ", spc)
        print("specificity for lymphocyte / sensitivity for other: ", sen)
        # We differentiate based on the dataset labels in this context.
class CellsClassifiDataset(Dataset):
    def __init__(self, path, cn, tn, cth, tth, featuretype="all"):
        data = cells_dataset(path, cellgraph_n=cn, tissuegraph_n=tn, cth=cth, tth=tth, ifdetector=False)
        data_list = data.create_dataset()
        data = np.vstack(data_list)
        self.ft = featuretype
        self.data = data[:, :-1].astype("float32")
        self.labels = data[:, -1].astype("float32")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        if self.ft == "ccg":
            x = self.data[index, :16]
        else:
            x = self.data[index, :]
        y = self.labels[index]
        return x, y
class LabelDataset(Dataset):
    def __init__(self, whole_data, labels):
        new_data = []
        new_labels = []
        for i in range(len(whole_data)):
            data, label = whole_data[i]
            if labels is None:
                new_data.append(data)
                new_labels.append(label)
            else:
                if label in labels:
                    new_data.append(data)
                    new_labels.append(label)
        self.data = new_data
        self.labels = new_labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y
    def count(self, ifprint=False):
        label_counts = {}
        for i in range(len(self.data)):
            label = self.labels[i]
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        if ifprint:
            for label, count in label_counts.items():
                print(f"Label: {label}, Count: {count}")
        return label_counts

if __name__ == "__main__":
    import sys
    arg = sys.argv
    task = arg[1]
    dataset_path = "./dataset/"
    model_path = "./models/ctg_pam.pth"
    if task == "pre":
        print("Start preprocessing.")
        dataset = cells_dataset(dataset_path, cellgraph_n=20, tissuegraph_n=2, cth=50, tth=5, ifdetector=True)
        dataset.cell_generate()
        dataset.tissue_generate()
        # dataset.save_png_map()
        print("Finish preprocessing.")
    elif task == "single":
        print("Start computing single-cell features and single-tissue features.")
        dataset = cells_dataset(dataset_path, cellgraph_n=20, tissuegraph_n=2, cth=50, tth=5, ifdetector=False)
        dataset.cell_feature_init()
        dataset.tissue_feature_init()
        dataset.distmatrix_init()
        print("Finish computing single-cell features and single-tissue features.")
    elif task == "ctg":
        print("Evolution of cell-cell features CCG and cell-tissue features by CTG.")
        dataset = cells_dataset(dataset_path, cellgraph_n=20, tissuegraph_n=2, cth=50, tth=5, ifdetector=False)
        dataset.cell_adjust_feature_init()
        dataset.label_init()
        print("Done.")
    elif task == "detect":
        print("Lymphocyte recognition and SS diagnosis.")
        dataset = cells_dataset(dataset_path, cellgraph_n=20, tissuegraph_n=2, cth=50, tth=5, ifdetector=False)
        dataset.cell_classification(model_path=model_path, visualization=True, mlp_n=50, feature_type="ctg", gpu="cuda:0")
    elif task == "all":
        dataset = cells_dataset(dataset_path, cellgraph_n=20, tissuegraph_n=2, cth=50, tth=5, ifdetector=True)
        dataset.cell_generate()
        dataset.tissue_generate()
        dataset.cell_feature_init()
        dataset.tissue_feature_init()
        dataset.distmatrix_init()
        dataset.cell_adjust_feature_init()
        dataset.label_init()
        dataset.cell_classification(model_path, visualization=True, mlp_n=50, feature_type="ctg", gpu="cuda:0")
    else:
        raise ValueError("The argument should be set in (pre, single, ctg, detect, all), here is %s."%arg[1])
        