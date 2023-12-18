import os.path


def make_LLVIP_dataset(path, mode):
    images = []
    assert os.path.isdir(path), '%s is not a valid directory' % path
    path_ir = os.path.join(path, 'infrared', mode)
    path_vi = os.path.join(path, 'visible', mode)
    for fname in sorted(os.listdir(path_vi)):
        fcode = fname[:-4]
        path_ir_img = os.path.join(path_ir, fname)
        print(path_ir_img + '\n')
        path_vi_img = os.path.join(path_vi, fname)
        print(path_vi_img + '\n')
        annotation_file = os.path.join(path, 'Annotations', fcode + '.xml')
        images.append({'A': path_vi_img, 'B': path_ir_img, "annotation_file": annotation_file})

    # if mode == 'train':
    #     pass
    # elif mode == 'test':
    #     pass
    
    return images


path = 'LLVIP'
path_ir = os.path.join(path, 'infrared')
path_ir_train = os.path.join(path_ir, 'train')
path_ir_test = os.path.join(path, 'infrared', 'test')
path_vi = os.path.join(path, 'visible')
path_vi_train = os.path.join(path_vi, 'train')
path_vi_test = os.path.join(path, 'visible' , 'test')
path_anno = os.path.join(path, 'Annotations')
make_LLVIP_dataset(path, 'test')





