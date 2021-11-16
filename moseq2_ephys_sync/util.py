import os
from glob import glob
import pdb

def remove_confounders(files, ext):
    if ext == 'txt':
        files = [f for f in files if ('depth_ts.txt' not in f)]
    return files

def find_file_through_glob_and_symlink(path, pattern):
    """Returns path to file found that matches pattern in path, or tries to follow symlink to raw data. Must only be one that matches!
    path: path to folder with data
    pattern: glob pattern, eg *.txt for arduino data
    """

    assert os.path.exists(path), f'Path {path} does not exist'
    files = glob(os.path.join(path,pattern))
    files = remove_confounders(files, pattern[-3:])

    if len(files) == 0:    
        # Find symlinked depth video (could eventually make dynamic but it's a pain)
        try_avi = glob(os.path.join(path,f'*depth.avi'))[0]
        if len(try_avi) == 0:
            try_mkv = glob(os.path.join(path,f'*depth.mkv'))[0]
            if len(try_mkv) == 0:
                raise RuntimeError(f'Could not find symlinked depth file in {base_path}')
            depth_path = try_mkv
        else:
            depth_path = try_avi
        
        # Follow it to find desired file
        sym_path = os.readlink(depth_path)
        containing_dir = os.path.dirname(sym_path)
        files = glob(os.path.join(containing_dir, pattern))
        files = remove_confounders(files, pattern[-3:])
        
        
    assert len(files) > 0, 'Found no files matching pattern'
    assert len(files) == 1, 'Found more than one file matching pattern!'

    return files[0]