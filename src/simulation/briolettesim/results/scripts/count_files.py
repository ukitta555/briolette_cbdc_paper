import os
import argparse

def count_files(directory):
    """
    Count the number of files in a directory and its subdirectories.
    
    Args:
        directory (str): Path to the directory to count files in
        
    Returns:
        tuple: (total_files, file_types_dict) where file_types_dict is a dictionary
               mapping file extensions to their counts
    """
    total_files = 0
    file_types = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            total_files += 1
            # Get file extension
            _, ext = os.path.splitext(file)
            ext = ext.lower() if ext else 'no_extension'
            file_types[ext] = file_types.get(ext, 0) + 1
    
    return total_files, file_types

def main():
    parser = argparse.ArgumentParser(description='Count files in a directory and its subdirectories.')
    parser.add_argument('directory', type=str, help='Directory to count files in')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    total_files, file_types = count_files(args.directory)
    
    print(f"\nDirectory: {args.directory}")
    print(f"Total number of files: {total_files}")
    print("\nBreakdown by file type:")
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{ext}: {count}")

if __name__ == "__main__":
    main() 