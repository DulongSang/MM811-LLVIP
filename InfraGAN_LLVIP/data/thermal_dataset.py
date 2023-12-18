import os.path
import torchvision.transforms as transforms
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.image_folder import make_thermal_dataset, is_image_file
from PIL import Image, ImageOps


def make_thermal_dataset_kaist(path=None, text_path=None):
    if path is None:
        path = '/cta/users/mehmet/rgbt-ped-detection/data/kaist-rgbt/images'
    if text_path is None:
        text_path = '/cta/users/mehmet/rgbt-ped-detection/data/scripts/imageSets/train-all-04.txt'
        text_path = '/cta/users/mehmet/rgbt-ped-detection/data/scripts/imageSets/test-all-20.txt'
    assert os.path.isfile(text_path), '%s is not a valid file' % text_path
    assert os.path.isdir(path), '%s is not a valid directory' % path
    images = []
    with open(text_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.split()[0]
        line = line.split('/')
        path_rgb = os.path.join(path, line[0])
        path_rgb = os.path.join(path_rgb, line[1])
        path_ir = os.path.join(path_rgb, 'lwir')
        path_ir = os.path.join(path_ir, line[2]+'.jpg')
        path_rgb = os.path.join(path_rgb, 'visible')
        path_rgb = os.path.join(path_rgb, line[2]+'.jpg')
        assert os.path.isfile(path_rgb), '%s is not a valid file' % path_rgb
        assert os.path.isfile(path_ir), '%s is not a valid file' % path_ir
        images.append({'A': path_rgb, 'B': path_ir, "annotation_file": os.path.join(path,
                                                                                    "..",
                                                                                    "annotations",
                                                                                    line[0],
                                                                                    line[1],
                                                                                    line[2]+'.txt')
                       })
    np.random.seed(12)
    np.random.shuffle(images)
    return images


def make_thermal_dataset_VEDAI(path):
    images = []
    assert os.path.isdir(path), '%s is not a valid directory' % path

    for fname in sorted(os.listdir(path)):
        if is_image_file(fname) and fname.endswith("co.png"):
            path_tv = os.path.join(path, fname)
            path_ir = fname[:-6] + "ir.png"
            path_ir = os.path.join(path, path_ir)
            annotation_file = os.path.join(path, "..", "Annotations1024", fname[:-7] + ".txt")
            images.append({'A': path_tv, 'B': path_ir, "annotation_file": annotation_file})
    return images


def make_thermal_dataset_LLVIP(path, mode):
    images = []
    assert os.path.isdir(path), '%s is not a valid directory' % path
    path_ir = os.path.join(path, 'infrared', mode)
    path_vi = os.path.join(path, 'visible', mode)
    for fname in sorted(os.listdir(path_vi)):
        fcode = fname[:-4]
        path_ir_img = os.path.join(path_ir, fname)
        #print(path_ir_img)
        path_vi_img = os.path.join(path_vi, fname)
        #print(path_vi_img)
        annotation_file = os.path.join(path, 'Annotations', fcode + '.xml')
        images.append({'A': path_vi_img, 'B': path_ir_img, "annotation_file": annotation_file, "img_name": fcode})

    # if mode == 'train':
    #     pass
    # elif mode == 'test':
    #     pass
    
    return images



class ThermalDataset(BaseDataset):
    def initialize(self, opt, mode='train'):
        print('ThermalDataset')
        mode = opt.phase
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        if opt.dataset_mode =='VEDAI':
            self.AB_paths = make_thermal_dataset_VEDAI(os.path.join(opt.dataroot, mode))
        elif opt.dataset_mode =='LLVIP':
            self.AB_paths = make_thermal_dataset_LLVIP(opt.dataroot, mode)
        elif opt.dataset_mode == 'KAIST':
            self.AB_paths = make_thermal_dataset_kaist(path=opt.dataroot, text_path=opt.text_path)
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        # AB_path = self.AB_paths[index]
        A_path = self.AB_paths[index]['A']
        B_path = self.AB_paths[index]['B']
        ann_path = self.AB_paths[index]['annotation_file']
        img_name = self.AB_paths[index]['img_name']
        A = Image.open(A_path).convert('RGB')
        #A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A.copy())
        
        B = Image.open(B_path)
        B = ImageOps.grayscale(B)
        #  B = B.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = transforms.ToTensor()(B.copy()).float()


        w_total = A.size(2)
        w = int(w_total)
        h = A.size(1)
        w_offset = max(0, (w - self.opt.fineSize - 1)//2)  # random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = max(0, (h - self.opt.fineSize - 1)//2)  # random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize([0.5], [0.5])(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path, "annotation_file": ann_path, 'img_name':img_name}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'ThermalDataset'
