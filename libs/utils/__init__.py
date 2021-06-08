import os
import json


__all__ = ['transfer_weights', 'PathManager', 'save_json']


def transfer_weights(src_model, dst_model):
    for idx, layer in enumerate(src_model.layers):
        layer_weights = layer.get_weights()
        if layer_weights == []:  # such as input, activation, concat ...
            continue
        try:
            dst_model.layers[idx].set_weights(layer_weights)
        except:
            print("* [Transfer Weights] Skipped Layer:", dst_model.layers[idx].name)


class PathManager:
    def __init__(self, path):
        path_no_ext, ext = os.path.splitext(path)
        basename_no_ext = os.path.basename(path_no_ext)
        self.path = path
        self.dir_path = os.path.dirname(path)
        self.basename_no_ext = basename_no_ext
        self.ext = ext
        self.fname = basename_no_ext + ext
        
    def get_parent_path(self, prefix_dir):
        prefix_dir = prefix_dir[:-1] if prefix_dir.endswith(os.sep) else prefix_dir
        prefix_dir_depth = len(prefix_dir.split(os.sep))
        parent_path = self.dir_path.split(os.sep)[prefix_dir_depth:]
        parent_path = os.path.join(*parent_path) if parent_path else ''
        parent_path = os.path.join(os.sep, parent_path)
        return parent_path

    @staticmethod
    def get_fpaths(src_dir, exts=['.png', '.jpg', '.jpeg']):
        fpaths = list()
        for current_path, _dirnames, fnames in os.walk(src_dir, followlinks=True):
            if not fnames:
                continue
                
            for fname in fnames:
                fname_ext = PathManager(path=fname).ext
                if not fname_ext in exts:
                    continue
                fpath = os.path.join(current_path, fname)
                fpaths.append(fpath)
        return fpaths


def save_json(obj, dst):
    with open(dst, 'wt', encoding='utf-8') as j:
        json.dump(obj, j, indent=4, ensure_ascii=False)
