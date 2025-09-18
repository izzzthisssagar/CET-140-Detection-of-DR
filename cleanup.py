import os
import shutil

def cleanup_project():
    print("Cleaning up project directory...")
    
    # Files to keep
    keep_files = {
        'simple_eye_gradcam.py',
        'main.py',
        'requirements.txt',
        'README.md',
        'cleanup.py'  # This file itself
    }
    
    # Directories to keep
    keep_dirs = {
        'results',
        'eye_gradcam_results'
    }
    
    # Files to remove
    files_to_remove = [
        'debug_gradcam.py',
        'debug_run.py',
        'download_sample_images.py',
        'eye_gradcam.py',
        'final_gradcam.py',
        'gradcam_final.py',
        'list_model_layers.py',
        'quick_gradcam.py',
        'run_gradcam.py',
        'run_gradcam_fixed.py',
        'simple_gradcam.py',
        'simple_test.py',
        'test_env.py',
        'test_gradcam_single.py',
        'test_model_image.py',
        'test_model_loading.py',
        'test_model_loading_v2.py',
        'test_output.txt',
        'test_image.jpg',
        'test_output',
        'gradcam_debug.log'
    ]
    
    # Remove files
    for file in files_to_remove:
        try:
            if os.path.isfile(file):
                os.remove(file)
                print(f"Removed file: {file}")
            elif os.path.isdir(file):
                shutil.rmtree(file)
                print(f"Removed directory: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")
    
    print("\nCleanup complete!")
    print("\nProject structure now contains:")
    for item in sorted(os.listdir('.')):
        if item in ['.git', '.gitignore']:
            continue
        if os.path.isfile(item) and item not in keep_files:
            print(f"WARNING: Unrecognized file: {item}")
        elif os.path.isdir(item) and item not in keep_dirs and not item.startswith('.'):
            print(f"WARNING: Unrecognized directory: {item}")
    
    print("\nTo update GitHub, run the following commands:")
    print("git add .")
    print('git commit -m "Clean up project and add Grad-CAM visualizations"')
    print("git push origin main")

if __name__ == "__main__":
    cleanup_project()
