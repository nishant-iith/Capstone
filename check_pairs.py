import os

def check_pairs(s_dir="1000/Stained_data", u_dir="1000/Unstained_data"):
    s_files = {f.replace("_stained.tif", "") for f in os.listdir(s_dir) if f.endswith("_stained.tif")}
    u_files = {f.replace("_unstained.tif", "") for f in os.listdir(u_dir) if f.endswith("_unstained.tif")}
    
    intersection = s_files.intersection(u_files)
    only_s = s_files - u_files
    only_u = u_files - s_files
    
    print(f"Total Stained files found: {len(s_files)}")
    print(f"Total Unstained files found: {len(u_files)}")
    print(f"Number of perfectly matched pairs: {len(intersection)}")
    
    if only_s:
        print(f"Stained files without Unstained pair: {len(only_s)}")
    if only_u:
        print(f"Unstained files without Stained pair: {len(only_u)}")

if __name__ == "__main__":
    check_pairs()
