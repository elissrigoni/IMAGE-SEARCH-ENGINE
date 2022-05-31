from tqdm import tqdm
import os
import shutil

move_in = "**********"      
general_directory = "********"  # dataset directory

for folders, subdirs, files in os.walk(scraped_path):
  i = 0
  for f in tqdm(files[1:]):
    
    cat = os.path.basename(folders)
    current_image_path = os.path.join(folders, f)
    
    # training
    if i < 300:
      target_folder = move_in + "/training/" + cat
      if not os.path.exists(target_folder):
        os.makedirs(target_folder)
      shutil.move(current_image_path, target_folder)

  
    # validation
    elif i >= 300  and i < 390:
      target_folder = move_in + "/new_validation/" + cat
      if not os.path.exists(target_folder):
        os.makedirs(target_folder)
      shutil.move(current_image_path, target_folder)

    # gallery
    elif i >= 390 and i < 442:
      target_folder = move_in + "/validation/gallery/" + cat
      if not os.path.exists(target_folder):
        os.makedirs(target_folder)
      shutil.move(current_image_path, target_folder)


      # query 
    elif i >= 442 and i < 453:
      target_folder = move_in + "/validation/query/" + cat
      if not os.path.exists(target_folder):
        os.makedirs(target_folder)
      shutil.move(current_image_path, target_folder)
    

    i += 1



diz_cat = {}

for folders, subdirs, files in os.walk(general_directory):
     cat = os.path.basename(folders)
     if cat != 'training':
       key = cat.split("(")[1][:-1]
       diz_cat[key] = cat


       
scraped_path = os.path.join(os.getcwd(), 'scraped_images_4')

for folders, subdirs, files in os.walk(scraped_path):
  i = 0
  for f in files[1:]:

    cat = os.path.basename(folders).split("_")[0]
    current_image_path = os.path.join(folders, f)

    if cat in diz_cat:
      category = diz_cat[cat]

      # training
      if i < 162:
        target_folder = move_in + "/training/" + category
        shutil.move(current_image_path, target_folder)

        # validation
      else:
        target_folder = move_in + "/new_validation/" + category
        if not os.path.exists(target_folder):
          os.makedirs(target_folder)
        shutil.move(current_image_path, target_folder)

    i += 1
