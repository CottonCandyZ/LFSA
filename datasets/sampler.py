from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K. 
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        # pid, image_id, img_path, caption
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        # random shuffle the ids in each pid
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                # Discard if last insufficient
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

class RandomIdentitySampler_Impove(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K. 
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        # pid, image_id, img_path, caption
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        self.img_index_dic = {}
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (pid, img_id, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            if pid in self.img_index_dic:
                self.img_index_dic[pid][img_id].append(index)
            else:
                img_dic = defaultdict(list)
                img_dic[img_id].append(index)
                self.img_index_dic[pid] = img_dic
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        img_index_dic = copy.deepcopy(self.img_index_dic)
        # random shuffle the ids in each pid
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
            else:
                idxs_ori = idxs
                idxs = []
                img_idxs = list(img_index_dic[pid].keys())
                while len(idxs) != len(idxs_ori):
                    random.shuffle(img_idxs)
                    for img_id in img_idxs:
                        d_list = img_index_dic[pid][img_id]
                        chosen_index = random.choice(d_list)
                        idxs.append(chosen_index)
                        d_list.remove(chosen_index)
                        if len(d_list) == 0:
                            img_idxs.remove(img_id)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                # Discard if last insufficient
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
    
class RandomIdentitySampler_Impove_v2(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K. 
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        # pid, image_id, img_path, caption
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        self.img_index_dic = {}
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (pid, img_id, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            if pid in self.img_index_dic:
                self.img_index_dic[pid][img_id].append(index)
            # first time
            else:
                img_dic = defaultdict(list)
                img_dic[img_id].append(index)
                self.img_index_dic[pid] = img_dic
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        img_index_dic = copy.deepcopy(self.img_index_dic)
        # random shuffle the ids in each pid
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
            else:
                idxs_ori = idxs
                idxs = []
                img_idxs = list(img_index_dic[pid].keys())
                
                while len(idxs) != len(idxs_ori):
                    chosen_img_idxs = np.random.choice(img_idxs, size=self.num_instances,replace=(len(img_idxs) < self.num_instances))
                    for img_id in chosen_img_idxs:
                        d_list = img_index_dic[pid][img_id]
                        if len(d_list) == 0:
                            continue
                        chosen_index = random.choice(d_list)
                        idxs.append(chosen_index)
                        d_list.remove(chosen_index)
                        if len(d_list) == 0:
                            img_idxs.remove(img_id)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                # Discard if last insufficient
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

class RandomIdentitySampler_More(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K. 
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        # pid, image_id, img_path, caption
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            if num % self.num_instances:
                self.length += self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        # random shuffle the ids in each pid
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            add_repeat = []
            if len(idxs) % self.num_instances:
                add_repeat = np.random.choice(idxs, size=(self.num_instances - len(idxs) % self.num_instances), replace=False).tolist()
            idxs.extend(add_repeat)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                # Discard if last insufficient
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
