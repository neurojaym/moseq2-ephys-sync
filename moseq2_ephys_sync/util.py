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

    # Check path exists
    assert os.path.exists(path), f'Path {path} does not exist'

    # Simply look for file
    files = glob(os.path.join(path,pattern))

    # Remove known confounders (eg we use '*.txt' for arduino files, but sometimes we also have 'depth_ts.txt')
    files = remove_confounders(files, pattern[-3:])


    # If no files found, maybe we have to follow a sym-linked depth-video back to the raw data directory
    if len(files) == 0:    
        try_avi = glob(os.path.join(path,f'*depth.avi'))[0]
        if len(try_avi) == 0:
            try_mkv = glob(os.path.join(path,f'*depth.mkv'))[0]
            if len(try_mkv) == 0:
                raise RuntimeError(f'Could not find symlinked depth file in {base_path}')
            depth_path = try_mkv
        else:
            depth_path = try_avi
        
        # Follow the symlink to find desired file
        sym_path = os.readlink(depth_path)
        containing_dir = os.path.dirname(sym_path)
        files = glob(os.path.join(containing_dir, pattern))
        files = remove_confounders(files, pattern[-3:])
        
    # Sanitize output
    
    assert len(files) > 0, 'Found no files matching pattern'
    assert len(files) == 1, 'Found more than one file matching pattern!'

    return files[0]
